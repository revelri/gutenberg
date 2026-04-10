<script lang="ts">
	import { onMount } from 'svelte';
	import { corpora, activeCorpus, conversations, activeConversation, messages, streamingContent, isStreaming, theme } from '$lib/stores';
	import { listCorpora, listConversations, getConversation, createConversation, streamMessage, createCorpus, uploadDocuments, streamIngestionStatus, getCorpus, deleteDocument, reingestDocument, renameConversation, deleteConversation } from '$lib/api';
	import type { Corpus, Conversation, Document, Mode, IngestionStatus, Message } from '$lib/types';
	import PdfViewer from '../components/PdfViewer.svelte';

	let queryInput = $state('');
	let selectedMode: Mode = $state('general');
	let showCorpusModal = $state(false);

	let corpusName = $state('');
	let corpusTags = $state('');
	let selectedFiles: File[] = $state([]);
	let isDragging = $state(false);
	let ingestProgress: IngestionStatus | null = $state(null);
	let fileInputEl: HTMLInputElement | undefined = $state(undefined);
	let isLoadingCorpora = $state(true);
	let conversationSearch = $state('');
	let pdfViewerState = $state<{ source: string; page: number; quote: string } | null>(null);
	let corpusDocuments: Document[] = $state([]);
	let isLoadingDocuments = $state(false);
	let deletingDocId = $state<string | null>(null);
	let reingestingDocId = $state<string | null>(null);
	let bookmarks: { quote: string; source: string; page: number; date: string }[] = $state([]);
	let docDeleteError = $state<string | null>(null);
	let docReingestError = $state<string | null>(null);
	let showDocList = $state(false);

	let editingConvId = $state<string | null>(null);
	let editTitle = $state('');
	let abortController = $state<AbortController | null>(null);
	let streamStartTime = $state<number | null>(null);
	let streamElapsed = $state(0);
	let streamTimer: ReturnType<typeof setInterval> | null = $state(null);
	let copyTooltip = $state<{ text: string; x: number; y: number } | null>(null);

	// Feature c: retry failed messages
	let failedMessageId = $state<string | null>(null);

	// Feature d: bulk delete conversations
	let selectMode = $state(false);
	let selectedIds = $state<Set<string>>(new Set());

	// Feature f: undo delete
	let pendingDelete = $state<{ conv: Conversation; timeout: ReturnType<typeof setTimeout> } | null>(null);

	let citationCount = $derived(
		$messages.filter(m => m.role === 'assistant')
			.reduce((sum, m) => sum + (m.content.match(/\[Source:/g) || []).length, 0)
	);
	let streamingWordCount = $derived(
		$streamingContent ? $streamingContent.split(/\s+/).filter(Boolean).length : 0
	);

	$effect(() => {
		if ($activeConversation) {
			document.title = $activeConversation.title || 'New conversation';
		} else {
			document.title = 'Gutenberg';
		}
	});

	// Feature e: session persistence
	function persistSession() {
		try {
			const data = {
				corpusId: $activeCorpus?.id || null,
				conversationId: $activeConversation?.id || null,
			};
			localStorage.setItem('gutenberg-session', JSON.stringify(data));
		} catch {}
	}

	async function restoreSession() {
		try {
			const raw = localStorage.getItem('gutenberg-session');
			if (!raw) return;
			const { corpusId, conversationId } = JSON.parse(raw);
			if (corpusId) {
				const corpus = $corpora.find(c => c.id === corpusId);
				if (corpus) {
					await selectCorpus(corpus);
					if (conversationId) {
						const conv = $conversations.find(c => c.id === conversationId);
						if (conv) {
							await selectConversation(conv);
						}
					}
				}
			}
		} catch {}
	}

	$effect(() => {
		const _c = $activeCorpus;
		const _v = $activeConversation;
		persistSession();
	});

	function openPdfViewer(source: string, page: number, quote: string) {
		pdfViewerState = { source, page, quote };
	}

	function loadBookmarks() {
		try {
			const raw = localStorage.getItem('gutenberg-bookmarks');
			bookmarks = raw ? JSON.parse(raw) : [];
		} catch {
			bookmarks = [];
		}
	}

	function addBookmark(quote: string, source: string, page: number) {
		bookmarks = [...bookmarks, { quote, source, page, date: new Date().toISOString() }];
		localStorage.setItem('gutenberg-bookmarks', JSON.stringify(bookmarks));
	}

	function printConversation() {
		const conv = $activeConversation;
		if (!conv) return;
		const w = window.open('', '_blank');
		if (!w) return;
		const msgsHtml = $messages.map(m => `
			<div style="margin-bottom:24px;padding:12px;border-left:3px solid ${m.role === 'user' ? '#8B4513' : '#D6D3CD'};">
				<div style="font-weight:600;font-size:12px;text-transform:uppercase;color:#A8A29E;margin-bottom:4px;">${m.role}</div>
				<div style="line-height:1.7;">${m.content.replace(/\n/g, '<br>')}</div>
			</div>
		`).join('');
		w.document.write(`<!DOCTYPE html><html><head><title>Gutenberg - ${conv.title || 'Conversation'}</title>
			<style>body{font-family:Georgia,serif;max-width:700px;margin:40px auto;padding:0 20px;color:#1C1917;}
			h1{font-size:24px;margin-bottom:4px;} .meta{font-size:13px;color:#A8A29E;margin-bottom:32px;}</style>
			</head><body><h1>${conv.title || 'Conversation'}</h1>
			<p class="meta">${conv.mode} mode &middot; ${new Date(conv.created_at).toLocaleDateString()}</p>
			${msgsHtml}</body></html>`);
		w.document.close();
		w.print();
	}

	async function loadCorpusDocuments() {
		const corpus = $activeCorpus;
		if (!corpus) return;
		isLoadingDocuments = true;
		corpusDocuments = [];
		try {
			const detail = await getCorpus(corpus.id);
			corpusDocuments = detail.documents || [];
		} catch (err) {
			corpusDocuments = [];
		} finally {
			isLoadingDocuments = false;
		}
	}

	async function handleDeleteDocument(docId: string) {
		const corpus = $activeCorpus;
		if (!corpus) return;
		deletingDocId = docId;
		docDeleteError = null;
		try {
			await deleteDocument(corpus.id, docId);
			await loadCorpusDocuments();
			const list = await listCorpora();
			corpora.set(list);
			const updated = list.find(c => c.id === corpus.id);
			if (updated) activeCorpus.set(updated);
		} catch (err) {
			docDeleteError = err instanceof Error ? err.message : 'Delete failed';
		} finally {
			deletingDocId = null;
		}
	}

	async function handleReingestDocument(docId: string) {
		const corpus = $activeCorpus;
		if (!corpus) return;
		reingestingDocId = docId;
		docReingestError = null;
		try {
			await reingestDocument(corpus.id, docId);
		} catch (err) {
			docReingestError = err instanceof Error ? err.message : 'Re-ingest failed';
		} finally {
			reingestingDocId = null;
		}
	}

	async function requestNotificationPermission() {
		if ('Notification' in window && Notification.permission === 'default') {
			await Notification.requestPermission();
		}
	}

	onMount(async () => {
		(window as any).__openPdf = (source: string, page: number, quote: string) => {
			openPdfViewer(source, page, quote);
		};
		(window as any).__addBookmark = (quote: string, source: string, page: number) => {
			addBookmark(quote, source, page);
		};
		(window as any).__copyCitation = (text: string, event: MouseEvent) => {
			navigator.clipboard.writeText(text);
			copyTooltip = { text: 'Copied!', x: event.clientX, y: event.clientY };
			setTimeout(() => { copyTooltip = null; }, 1500);
		};
		loadBookmarks();
		try {
			const list = await listCorpora();
			corpora.set(list);
			if (list.length > 0) {
				await selectCorpus(list[0]);
				await restoreSession();
			}
		} finally {
			isLoadingCorpora = false;
		}
	});

	async function selectCorpus(corpus: Corpus) {
		activeCorpus.set(corpus);
		const convs = await listConversations(corpus.id);
		conversations.set(convs);
		activeConversation.set(null);
		messages.set([]);
	}

	async function selectConversation(conv: Conversation) {
		activeConversation.set(conv);
		selectedMode = conv.mode;
		const detail = await getConversation(conv.id);
		messages.set(detail.messages);
	}

	async function newConversation() {
		const corpus = $activeCorpus;
		if (!corpus) return;
		const conv = await createConversation(corpus.id, selectedMode);
		conversations.update(c => [conv, ...c]);
		activeConversation.set(conv);
		messages.set([]);
	}

	async function retryFailedMessage(failedMsgId: string) {
		const failedMsg = $messages.find(m => m.id === failedMsgId);
		if (!failedMsg) return;
		failedMessageId = null;
		messages.update(m => m.filter(msg => msg.id !== failedMsgId));
		const content = failedMsg.content;
		const conv = $activeConversation;
		if (!conv) return;

		messages.update(m => [...m, {
			id: crypto.randomUUID(),
			conversation_id: conv.id,
			role: 'user' as const,
			content,
			metadata_json: '{}',
			created_at: new Date().toISOString(),
		}]);

		isStreaming.set(true);
		streamingContent.set('');
		abortController = new AbortController();
		streamStartTime = Date.now();
		streamElapsed = 0;
		streamTimer = setInterval(() => {
			if (streamStartTime) streamElapsed = (Date.now() - streamStartTime) / 1000;
		}, 500);

		try {
			for await (const chunk of streamMessage(conv.id, content, undefined, abortController.signal)) {
				streamingContent.update(s => s + chunk);
			}

			const fullResponse = $streamingContent;
			messages.update(m => [...m, {
				id: crypto.randomUUID(),
				conversation_id: conv.id,
				role: 'assistant' as const,
				content: fullResponse,
				metadata_json: '{}',
				created_at: new Date().toISOString(),
			}]);
			streamingContent.set('');
		} catch {
			const currentMsgs = $messages;
			const lastUserMsg = [...currentMsgs].reverse().find(m => m.role === 'user');
			if (lastUserMsg) failedMessageId = lastUserMsg.id;
		} finally {
			isStreaming.set(false);
			abortController = null;
			if (streamTimer) { clearInterval(streamTimer); streamTimer = null; }
			streamStartTime = null;
		}
	}

	async function sendMessage() {
		const conv = $activeConversation;
		if (!conv || !queryInput.trim() || $isStreaming) return;

		const content = queryInput.trim();
		queryInput = '';

		const userMsgId = crypto.randomUUID();

		messages.update(m => [...m, {
			id: userMsgId,
			conversation_id: conv.id,
			role: 'user' as const,
			content,
			metadata_json: '{}',
			created_at: new Date().toISOString(),
		}]);

		isStreaming.set(true);
		streamingContent.set('');
		abortController = new AbortController();
		streamStartTime = Date.now();
		streamElapsed = 0;
		streamTimer = setInterval(() => {
			if (streamStartTime) streamElapsed = (Date.now() - streamStartTime) / 1000;
		}, 500);

		try {
			for await (const chunk of streamMessage(conv.id, content, undefined, abortController.signal)) {
				streamingContent.update(s => s + chunk);
			}

			const fullResponse = $streamingContent;
			messages.update(m => [...m, {
				id: crypto.randomUUID(),
				conversation_id: conv.id,
				role: 'assistant' as const,
				content: fullResponse,
				metadata_json: '{}',
				created_at: new Date().toISOString(),
			}]);
			streamingContent.set('');
		} catch {
			failedMessageId = userMsgId;
		} finally {
			isStreaming.set(false);
			abortController = null;
			if (streamTimer) { clearInterval(streamTimer); streamTimer = null; }
			streamStartTime = null;
		}
	}

	function toggleSelectMode() {
		selectMode = !selectMode;
		if (!selectMode) {
			selectedIds = new Set();
		}
	}

	function toggleConvSelection(id: string) {
		const next = new Set(selectedIds);
		if (next.has(id)) {
			next.delete(id);
		} else {
			next.add(id);
		}
		selectedIds = next;
	}

	async function deleteSelected() {
		if (selectedIds.size === 0) return;
		const ids = [...selectedIds];
		await Promise.all(ids.map(id => deleteConversation(id)));
		selectedIds = new Set();
		selectMode = false;
		const corpus = $activeCorpus;
		if (corpus) {
			const convs = await listConversations(corpus.id);
			conversations.set(convs);
			if ($activeConversation && !convs.find(c => c.id === $activeConversation!.id)) {
				activeConversation.set(null);
				messages.set([]);
			}
		}
	}

	function deleteConversationWithUndo(conv: Conversation) {
		conversations.update(c => c.filter(x => x.id !== conv.id));
		if ($activeConversation?.id === conv.id) {
			activeConversation.set(null);
			messages.set([]);
		}
		const timeout = setTimeout(async () => {
			pendingDelete = null;
			try {
				await deleteConversation(conv.id);
			} catch {}
		}, 30000);
		pendingDelete = { conv, timeout };
	}

	function undoDelete() {
		if (!pendingDelete) return;
		clearTimeout(pendingDelete.timeout);
		conversations.update(c => [pendingDelete!.conv, ...c]);
		pendingDelete = null;
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			sendMessage();
		}
		if (e.ctrlKey && e.key === 'Enter') {
			e.preventDefault();
			sendMessage();
		}
	}

	function handleGlobalKeydown(e: KeyboardEvent) {
		if (e.ctrlKey && !e.shiftKey && e.key === 'n') {
			e.preventDefault();
			newConversation();
		}
		if (e.ctrlKey && e.shiftKey && e.key === 'N') {
			e.preventDefault();
			showCorpusModal = true;
		}
		if (e.key === 'Escape') {
			if (showCorpusModal) {
				showCorpusModal = false;
			}
		}
	}

	function toggleTheme() {
		theme.update(t => t === 'light' ? 'dark' : 'light');
	}

	function startRename(conv: Conversation) {
		editingConvId = conv.id;
		editTitle = conv.title || 'New conversation';
	}

	function cancelRename() {
		editingConvId = null;
	}

	async function finishRename(conv: Conversation) {
		const title = editTitle.trim();
		if (title && title !== (conv.title || 'New conversation')) {
			await renameConversation(conv.id, title);
			conversations.update(convs =>
				convs.map(c => c.id === conv.id ? { ...c, title } : c)
			);
			if ($activeConversation?.id === conv.id) {
				activeConversation.update(c => c ? { ...c, title } : c);
			}
		}
		editingConvId = null;
	}

	function stopStreaming() {
		abortController?.abort();
	}

	function copyToClipboard(text: string) {
		navigator.clipboard.writeText(text);
	}

	function formatResponse(text: string): string {
		return text
			.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
			.replace(/\n/g, '<br>')
			.replace(/&lt;blockquote&gt;/g, '<blockquote>').replace(/&lt;\/blockquote&gt;/g, '</blockquote>')
			.replace(/&quot;([^&]{20,}?)&quot;/g, '<blockquote>&quot;$1&quot;</blockquote>')
			.replace(/"([^"]{20,}?)"/g, '<blockquote>"$1"</blockquote>')
			.replace(
				/\[Source:\s*([^,\]]+),\s*p\.\s*(\d+)\]/g,
				'<span class="citation-badge" style="cursor:pointer;" onclick="window.__copyCitation(this.textContent, event)">[Source: $1, p. $2]</span>' +
				'<button class="btn-ghost view-pdf-btn" data-source="$1" data-page="$2" onclick="window.__openPdf(this.dataset.source, parseInt(this.dataset.page), this.dataset.source)">View</button>' +
				'<button class="btn-ghost bookmark-btn" data-source="$1" data-page="$2" onclick="window.__addBookmark(\'\', this.dataset.source, parseInt(this.dataset.page))" title="Bookmark">&#9734;</button>'
			)
			.replace(
				/\[Source:\s*([^\]]+)\]/g,
				'<span class="citation-badge" style="cursor:pointer;" onclick="window.__copyCitation(this.textContent, event)">[Source: $1]</span>'
			);
	}

	const modeLabels: Record<Mode, string> = {
		exact: 'Exact Citation',
		general: 'General',
		exhaustive: 'Exhaustive',
		precis: 'Precis',
	};

	const modePlaceholders: Record<Mode, string> = {
		exact: 'Find the exact passage: "..."',
		general: 'Ask a question about the corpus...',
		exhaustive: "Find every instance of 'concept' across all works...",
		precis: 'Trace the evolution of concept X from Work A to Work B...',
	};

	function handleFileSelect(e: Event) {
		const input = e.target as HTMLInputElement;
		if (input.files) {
			for (const f of input.files) {
				if (!selectedFiles.find(x => x.name === f.name)) {
					selectedFiles = [...selectedFiles, f];
				}
			}
		}
		input.value = '';
	}

	function removeFile(index: number) {
		selectedFiles = selectedFiles.filter((_, i) => i !== index);
	}

	function handleDrop(e: DragEvent) {
		e.preventDefault();
		isDragging = false;
		if (e.dataTransfer?.files) {
			for (const f of e.dataTransfer.files) {
				if (!selectedFiles.find(x => x.name === f.name)) {
					selectedFiles = [...selectedFiles, f];
				}
			}
		}
	}

	function handleDragOver(e: DragEvent) {
		e.preventDefault();
		isDragging = true;
	}

	function handleDragLeave() {
		isDragging = false;
	}

	function resetModal() {
		corpusName = '';
		corpusTags = '';
		selectedFiles = [];
		isDragging = false;
		ingestProgress = null;
	}

	function formatFileSize(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
	}

	async function buildCorpus() {
		if (!corpusName.trim() || selectedFiles.length === 0) return;

		await requestNotificationPermission();

		let corpusIdForNotif = '';

		try {
			const { id } = await createCorpus(corpusName.trim(), corpusTags.trim());
			corpusIdForNotif = id;

			ingestProgress = {
				job_id: '',
				total_files: selectedFiles.length,
				completed_files: 0,
				current_file: null,
				current_step: 'uploading',
				status: 'running',
				error: null,
			};
			await uploadDocuments(id, selectedFiles);

			const es = streamIngestionStatus(id, (progress) => {
				ingestProgress = progress;
			});

			const checkDone = setInterval(() => {
				if (es.readyState === EventSource.CLOSED || ingestProgress?.status === 'done' || ingestProgress?.status === 'failed') {
					clearInterval(checkDone);
					es.close();
					if (document.visibilityState === 'hidden' && 'Notification' in window && Notification.permission === 'granted') {
						const statusMsg = ingestProgress?.status === 'done'
							? `Corpus "${corpusName.trim()}" ingestion complete.`
							: `Corpus "${corpusName.trim()}" ingestion failed.`;
						new Notification('Gutenberg', { body: statusMsg });
					}
					showCorpusModal = false;
					resetModal();
					listCorpora().then(list => {
						corpora.set(list);
						if (list.length > 0 && !$activeCorpus) {
							selectCorpus(list[0]);
						}
					});
				}
			}, 500);
		} catch (err) {
			console.error('Corpus creation failed:', err);
			ingestProgress = {
				job_id: '',
				total_files: 0,
				completed_files: 0,
				current_file: null,
				current_step: null,
				status: 'failed',
				error: err instanceof Error ? err.message : 'Unknown error',
			};
		}
	}
</script>

<div class="app-shell" onkeydown={handleGlobalKeydown}>
	<!-- Top Bar -->
	<div class="topbar">
		<h4 style="font-family: var(--font-display); font-weight: 400; font-size: 1.25rem;">Gutenberg</h4>
		<div style="flex: 1;"></div>

		{#if $corpora.length > 0}
			<select
				value={$activeCorpus?.id || ''}
				onchange={(e) => {
					const c = $corpora.find(x => x.id === (e.target as HTMLSelectElement).value);
					if (c) selectCorpus(c);
				}}
			>
				{#each $corpora as corpus}
					<option value={corpus.id}>{corpus.name}</option>
				{/each}
			</select>
		{/if}

		{#if $corpora.length === 0 && !isLoadingCorpora}
			<p style="color: var(--color-ash); font-size: 13px; padding: var(--space-md); text-align: center;">
				Welcome to Gutenberg.<br>Create your first corpus to get started.
			</p>
		{/if}

		<button class="btn-secondary" onclick={() => showCorpusModal = true}>+ New Corpus</button>
		<button class="btn-ghost" onclick={toggleTheme}>
			{$theme === 'light' ? '◐' : '◑'}
		</button>
	</div>

	<!-- Left Pane: Conversations -->
	<div class="left-pane">
		<div style="padding: var(--space-md); display: flex; align-items: center; gap: var(--space-sm);">
			<span class="section-label" style="flex: 1;">Conversations</span>
			{#if $conversations.length > 0}
				<button class="btn-ghost select-toggle-btn" onclick={toggleSelectMode}>
					{selectMode ? 'Cancel' : 'Select'}
				</button>
			{/if}
			<button class="btn-primary" style="padding: 4px 12px; font-size: 13px;" onclick={newConversation}>+</button>
		</div>

		{#if $conversations.length > 0}
			<div style="padding: 0 var(--space-sm) var(--space-sm);">
				<input type="text" placeholder="Search conversations..." bind:value={conversationSearch}
					style="width: 100%; padding: 6px 10px; font-size: 13px; border: 1px solid var(--color-parchment); border-radius: var(--radius-sm); background: transparent; font-family: var(--font-body); color: var(--color-ink);" />
			</div>
		{/if}

		<div style="padding: 0 var(--space-sm);">
			{#if $conversations.length === 0 && isLoadingCorpora}
				<div class="loading-text">
					<span class="spinner"></span>
					Loading...
				</div>
			{:else}
				{#each $conversations.filter(c => !conversationSearch || (c.title || '').toLowerCase().includes(conversationSearch.toLowerCase())) as conv}
					<div class="conv-row">
						{#if selectMode}
							<input
								type="checkbox"
								checked={selectedIds.has(conv.id)}
								onchange={() => toggleConvSelection(conv.id)}
								class="conv-checkbox"
							/>
						{/if}
						<button
							class="conv-item"
							class:active={$activeConversation?.id === conv.id}
							onclick={() => { if (!selectMode && editingConvId !== conv.id) selectConversation(conv); }}
						>
							{#if editingConvId === conv.id}
								<input
									type="text"
									bind:value={editTitle}
									class="rename-input"
									onclick={(e) => e.stopPropagation()}
									onkeydown={(e) => {
										if (e.key === 'Enter') { e.preventDefault(); finishRename(conv); }
										if (e.key === 'Escape') { e.preventDefault(); cancelRename(); }
										e.stopPropagation();
									}}
									onblur={() => finishRename(conv)}
								/>
							{:else}
								<span class="conv-title" onclick={(e) => { e.stopPropagation(); if (!selectMode) startRename(conv); }}>{conv.title || 'New conversation'}</span>
								<span class="mode-badge">{conv.mode}</span>
								{#if $activeConversation?.id === conv.id && citationCount > 0}
									<span class="citation-count-badge">{citationCount}</span>
								{/if}
							{/if}
						</button>
						{#if !selectMode}
							<button class="btn-ghost conv-delete-btn" onclick={() => deleteConversationWithUndo(conv)} title="Delete">&times;</button>
						{/if}
					</div>
				{/each}

				{#if selectMode && selectedIds.size > 0}
					<button class="btn-primary delete-selected-btn" onclick={deleteSelected}>
						Delete Selected ({selectedIds.size})
					</button>
				{/if}

				{#if pendingDelete}
					<div class="undo-toast">
						<span>Deleted.</span>
						<button class="btn-ghost" onclick={undoDelete}>Undo?</button>
					</div>
				{/if}

				{#if $conversations.length === 0}
					<p style="color: var(--color-ash); font-size: 13px; padding: var(--space-md); text-align: center;">
						No conversations yet.<br>Click + to start.
					</p>
				{/if}
			{/if}
		</div>
	</div>

	<!-- Center Pane: Results -->
	<div class="center-pane">
		{#if $messages.length === 0 && !$streamingContent}
			<div class="empty-state">
				<h2>Gutenberg</h2>
				<p>Semantic citation retrieval for scholarly corpora.</p>
				{#if $activeCorpus}
					<p class="data" style="color: var(--color-ash); margin-top: var(--space-md);">
						{$activeCorpus.name} &mdash; {$activeCorpus.chunk_count.toLocaleString()} chunks, {$activeCorpus.document_count} documents
					</p>
					<button class="btn-ghost" style="font-size: 12px; margin-top: var(--space-sm);" onclick={() => { showDocList = !showDocList; if (showDocList) loadCorpusDocuments(); }}>
						{showDocList ? 'Hide' : 'View'} documents
					</button>
				{/if}
				{#if showDocList && $activeCorpus}
					<div class="doc-list-card">
						{#if isLoadingDocuments}
							<div class="loading-text"><span class="spinner"></span> Loading...</div>
						{:else if corpusDocuments.length === 0}
							<p style="color: var(--color-ash); font-size: 13px;">No documents found.</p>
						{:else}
							{#each corpusDocuments as doc}
								<div class="doc-item">
									<div style="flex: 1; text-align: left;">
										<span style="font-weight: 600; font-size: 13px;">{doc.filename}</span>
										<span style="color: var(--color-ash); font-size: 12px; margin-left: var(--space-sm);">{doc.chunks} chunks</span>
										{#if doc.error}
											<span style="color: var(--color-error); font-size: 12px; margin-left: var(--space-sm);">{doc.error}</span>
										{/if}
									</div>
									<button
										class="btn-ghost doc-action-btn"
										disabled={reingestingDocId === doc.id}
										onclick={() => handleReingestDocument(doc.id)}
										title="Re-ingest"
									>
										{#if reingestingDocId === doc.id}<span class="spinner" style="width:12px;height:12px;"></span>{:else}&#8635;{/if}
									</button>
									<button
										class="btn-ghost doc-action-btn"
										disabled={deletingDocId === doc.id}
										onclick={() => handleDeleteDocument(doc.id)}
										title="Delete"
									>
										{#if deletingDocId === doc.id}<span class="spinner" style="width:12px;height:12px;"></span>{:else}&times;{/if}
									</button>
								</div>
							{/each}
							{#if docDeleteError}
								<p style="color: var(--color-error); font-size: 12px; margin-top: var(--space-sm);">{docDeleteError}</p>
							{/if}
							{#if docReingestError}
								<p style="color: var(--color-error); font-size: 12px; margin-top: var(--space-sm);">{docReingestError}</p>
							{/if}
						{/if}
					</div>
				{/if}
				{#if $activeConversation}
					<div style="margin-top: var(--space-lg); text-align: left; max-width: 400px;">
						{#if $activeConversation.mode === 'exhaustive'}
							<p class="tip" style="color: var(--color-ash); font-size: 13px;">
								Tip: put the concept in single quotes for best results
							</p>
						{/if}
					</div>
				{/if}
			</div>
		{:else}
			<div class="messages-container">
				<div class="messages-toolbar">
					{#if bookmarks.length > 0}
						<span class="bookmark-count" title="{bookmarks.length} bookmarked citations">&#9734; {bookmarks.length}</span>
					{/if}
					<button class="btn-ghost" style="font-size: 13px; padding: 2px 8px;" onclick={printConversation} disabled={!$activeConversation}>
						Print
					</button>
				</div>
				{#each $messages as msg}
					<div class="message" class:user={msg.role === 'user'} class:assistant={msg.role === 'assistant'}>
						{#if msg.role === 'user'}
							<div class="message-label section-label">Query</div>
							<p style="font-weight: 600;">{msg.content}</p>
							{#if failedMessageId === msg.id}
								<button class="btn-ghost retry-btn" onclick={() => retryFailedMessage(msg.id)}>Retry</button>
							{/if}
						{:else}
							<div class="message-label section-label">Response</div>
							<div class="response-content">{@html formatResponse(msg.content)}</div>
							<button class="btn-ghost copy-msg-btn" onclick={() => copyToClipboard(msg.content)}>Copy</button>
						{/if}
					</div>
					<hr />
				{/each}

				{#if $streamingContent}
					<div class="message assistant">
						<div class="message-label section-label">Response</div>
						<div class="response-content streaming">{@html formatResponse($streamingContent)}</div>
						{#if streamStartTime}
							<div class="streaming-stats">{streamingWordCount} words &middot; {streamElapsed.toFixed(1)}s</div>
						{/if}
					</div>
				{/if}
			</div>
		{/if}
	</div>

	<!-- Right Pane: Input -->
	<div class="right-pane">
		<div style="padding: var(--space-md); border-bottom: 1px solid var(--color-parchment);">
			<span class="section-label">Mode</span>
			<select bind:value={selectedMode} disabled={$activeConversation !== null} style="width: 100%; margin-top: var(--space-xs);">
				{#each (['exact', 'general', 'exhaustive', 'precis'] as const) as mode}
					<option value={mode}>{modeLabels[mode]}</option>
				{/each}
			</select>
		</div>

		<div style="flex: 1; overflow-y: auto; padding: var(--space-md);">
			{#if !$activeConversation}
				<p style="color: var(--color-ash); font-size: 13px;">
					Select or create a conversation to begin querying.
				</p>
			{/if}
		</div>

		<div style="padding: var(--space-md); border-top: 1px solid var(--color-parchment);">
			<textarea
				bind:value={queryInput}
				onkeydown={handleKeydown}
				placeholder={modePlaceholders[selectedMode]}
				disabled={!$activeConversation || $isStreaming}
				rows={3}
				style="width: 100%; margin-bottom: var(--space-sm);"
			></textarea>
			{#if $isStreaming}
				<button
					class="btn-secondary"
					style="width: 100%;"
					onclick={stopStreaming}
				>
					Stop
				</button>
			{:else}
				<button
					class="btn-primary"
					style="width: 100%;"
					onclick={sendMessage}
					disabled={!$activeConversation || !queryInput.trim()}
				>
					Send
				</button>
			{/if}
		</div>
	</div>
</div>

{#if copyTooltip}
	<div class="copy-tooltip" style="left: {copyTooltip.x}px; top: {copyTooltip.y - 30}px;">{copyTooltip.text}</div>
{/if}

{#if showCorpusModal}
	<div class="modal-overlay" role="dialog" onclick={() => { resetModal(); showCorpusModal = false; }}>
		<div class="modal-content card" onclick={(e) => e.stopPropagation()}>
			<h3>New Corpus</h3>
			<p style="color: var(--color-stone); margin: var(--space-sm) 0 var(--space-md);">
				Create a corpus project and upload documents for ingestion.
			</p>

			<label class="field-label">
				Name
				<input type="text" bind:value={corpusName} placeholder="e.g. Deleuze Critical Theory" style="width: 100%; margin-top: 4px;" />
			</label>

			<label class="field-label" style="margin-top: var(--space-md); display: block;">
				Tags
				<input type="text" bind:value={corpusTags} placeholder="philosophy, deleuze, poststructuralism" style="width: 100%; margin-top: 4px;" />
			</label>

			<div
				class="drop-zone"
				class:dragging={isDragging}
				style="margin-top: var(--space-md);"
				ondrop={handleDrop}
				ondragover={handleDragOver}
				ondragleave={handleDragLeave}
			>
				<p>Drag & drop PDF, EPUB, or DOCX files here</p>
				<p style="color: var(--color-ash); font-size: 13px; margin: var(--space-xs) 0;">or</p>
				<button class="btn-secondary" onclick={() => fileInputEl?.click()}>Browse Files</button>
				<input
					bind:this={fileInputEl}
					type="file"
					multiple
					accept=".pdf,.epub,.docx,.txt,.md"
					onchange={handleFileSelect}
					style="display: none;"
				/>
			</div>

			{#if selectedFiles.length > 0}
				<div class="file-list">
					{#each selectedFiles as file, i}
						<div class="file-item">
							<span style="flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{file.name}</span>
							<span style="color: var(--color-ash); font-size: 12px; white-space: nowrap;">{formatFileSize(file.size)}</span>
							<button onclick={() => removeFile(i)}>&times;</button>
						</div>
					{/each}
				</div>
			{/if}

			{#if ingestProgress}
				<div class="progress-bar-container">
					<div class="progress-bar">
						<div
							class="progress-bar-fill"
							style="width: {ingestProgress.total_files > 0
								? Math.round((ingestProgress.completed_files / ingestProgress.total_files) * 100)
								: 0}%"
						></div>
					</div>
					<p class="progress-text">
						{#if ingestProgress.status === 'failed'}
							Failed: {ingestProgress.error || 'Unknown error'}
						{:else if ingestProgress.status === 'done'}
							Complete - {ingestProgress.completed_files}/{ingestProgress.total_files} files processed
						{:else}
							{ingestProgress.completed_files}/{ingestProgress.total_files} files
							{#if ingestProgress.current_file}
								&mdash; processing {ingestProgress.current_file}
							{/if}
						{/if}
					</p>
				</div>
			{/if}

			<div style="display: flex; gap: var(--space-sm); margin-top: var(--space-lg); justify-content: flex-end;">
				<button class="btn-secondary" onclick={() => { resetModal(); showCorpusModal = false; }}>Cancel</button>
				<button
					class="btn-primary"
					onclick={buildCorpus}
					disabled={!corpusName.trim() || selectedFiles.length === 0 || (ingestProgress?.status === 'running')}
				>
					{ingestProgress?.status === 'running' ? 'Building...' : 'Build'}
				</button>
			</div>
		</div>
	</div>
{/if}

{#if pdfViewerState}
	<PdfViewer
		source={pdfViewerState.source}
		page={pdfViewerState.page}
		quote={pdfViewerState.quote}
		onclose={() => pdfViewerState = null}
	/>
{/if}

<style>
	.empty-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		height: 100%;
		text-align: center;
		color: var(--color-stone);
	}
	.empty-state h2 {
		font-size: 3rem;
		color: var(--color-primary);
		margin-bottom: var(--space-sm);
	}

	.conv-row {
		display: flex;
		align-items: center;
		gap: var(--space-xs);
	}

	.conv-item {
		display: flex;
		align-items: center;
		gap: var(--space-sm);
		width: 100%;
		padding: var(--space-sm) var(--space-md);
		background: transparent;
		border: none;
		border-radius: var(--radius-sm);
		cursor: pointer;
		text-align: left;
		font-family: var(--font-body);
		font-size: 14px;
		color: var(--color-ink);
		transition: background var(--duration-micro) ease-out;
	}
	.conv-item:hover { background: var(--color-primary-faint); }
	.conv-item.active { background: var(--color-primary-faint); border-left: 2px solid var(--color-primary); }
	.conv-title { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

	.conv-checkbox {
		flex-shrink: 0;
		width: 16px;
		height: 16px;
		cursor: pointer;
		margin-left: var(--space-sm);
		accent-color: var(--color-primary);
	}

	.conv-delete-btn {
		flex-shrink: 0;
		font-size: 16px;
		padding: 2px 6px;
		text-decoration: none;
		line-height: 1;
		opacity: 0;
		transition: opacity var(--duration-micro) ease-out;
	}
	.conv-row:hover .conv-delete-btn { opacity: 1; }

	.select-toggle-btn {
		font-size: 12px;
		padding: 2px 8px;
		text-decoration: none;
	}

	.delete-selected-btn {
		width: 100%;
		margin-top: var(--space-sm);
		padding: var(--space-xs) var(--space-sm);
		font-size: 12px;
		background: var(--color-error);
		border-radius: var(--radius-sm);
	}

	.undo-toast {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: var(--space-sm);
		padding: var(--space-sm);
		margin-top: var(--space-sm);
		background: var(--color-parchment);
		border-radius: var(--radius-sm);
		font-size: 13px;
		color: var(--color-stone);
	}

	.retry-btn {
		font-size: 12px;
		padding: 2px 8px;
		margin-top: var(--space-xs);
		text-decoration: none;
	}

	.messages-container { max-width: 800px; margin: 0 auto; }
	.message { margin-bottom: var(--space-md); }
	.message-label { margin-bottom: var(--space-xs); }
	.response-content { line-height: 1.7; }
	.streaming { opacity: 0.85; }

	.modal-overlay {
		position: fixed;
		inset: 0;
		background: rgba(0,0,0,0.4);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 100;
	}
	.modal-content { width: 520px; max-height: 80vh; overflow-y: auto; padding: var(--space-lg); }

	.field-label { font-size: 14px; font-weight: 600; }

	.drop-zone {
		border: 2px dashed var(--color-parchment);
		border-radius: var(--radius-md);
		padding: var(--space-xl);
		text-align: center;
		color: var(--color-stone);
		transition: border-color var(--duration-short) ease-out;
	}
	.drop-zone:hover { border-color: var(--color-primary); }
	.drop-zone.dragging { border-color: var(--color-primary) !important; background: var(--color-primary-faint); }

	.file-list {
		margin-top: var(--space-sm);
		max-height: 200px;
		overflow-y: auto;
	}
	.file-item {
		display: flex;
		align-items: center;
		gap: var(--space-sm);
		padding: var(--space-xs) var(--space-sm);
		border-bottom: 1px solid var(--color-parchment);
		font-size: 13px;
	}
	.file-item button {
		background: none;
		border: none;
		color: var(--color-ash);
		cursor: pointer;
		padding: 0 4px;
		font-size: 16px;
	}
	.file-item button:hover { color: var(--color-primary); }

	.progress-bar-container { margin-top: var(--space-md); }
	.progress-bar {
		width: 100%;
		height: 6px;
		background: var(--color-parchment);
		border-radius: 3px;
		overflow: hidden;
	}
	.progress-bar-fill {
		height: 100%;
		background: var(--color-primary);
		transition: width 0.3s ease;
	}
	.progress-text {
		font-size: 12px;
		color: var(--color-ash);
		margin-top: var(--space-xs);
	}

	.view-pdf-btn {
		font-size: 11px;
		padding: 1px 6px;
		margin-left: 4px;
		text-decoration: underline;
		cursor: pointer;
	}

	.spinner {
		display: inline-block;
		width: 16px;
		height: 16px;
		border: 2px solid var(--color-parchment);
		border-top-color: var(--color-primary);
		border-radius: 50%;
		animation: spin 0.6s linear infinite;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	.loading-text {
		display: flex;
		align-items: center;
		gap: var(--space-sm);
		color: var(--color-ash);
		font-size: 13px;
		padding: var(--space-md);
		justify-content: center;
	}

	.messages-toolbar {
		display: flex;
		align-items: center;
		justify-content: flex-end;
		gap: var(--space-sm);
		margin-bottom: var(--space-md);
	}

	.bookmark-count {
		font-family: var(--font-data);
		font-size: 12px;
		color: var(--color-primary);
		background: var(--color-primary-faint);
		padding: 2px 8px;
		border-radius: var(--radius-sm);
	}

	.doc-list-card {
		margin-top: var(--space-md);
		width: 100%;
		max-width: 500px;
		text-align: left;
		background: var(--color-surface);
		border: 1px solid var(--color-parchment);
		border-radius: var(--radius-md);
		padding: var(--space-sm);
	}

	.doc-item {
		display: flex;
		align-items: center;
		gap: var(--space-sm);
		padding: var(--space-xs) var(--space-sm);
		border-bottom: 1px solid var(--color-parchment);
		font-size: 13px;
	}
	.doc-item:last-child { border-bottom: none; }

	.doc-action-btn {
		font-size: 14px;
		padding: 2px 6px;
		text-decoration: none;
		line-height: 1;
	}

	.bookmark-btn {
		font-size: 13px;
		padding: 0 4px;
		text-decoration: none;
		cursor: pointer;
	}
	.bookmark-btn:hover {
		color: var(--color-primary);
	}

	.rename-input {
		flex: 1;
		padding: 2px 6px;
		font-size: 14px;
		font-family: var(--font-body);
		border: 1px solid var(--color-primary);
		border-radius: var(--radius-sm);
		background: var(--color-surface);
		color: var(--color-ink);
		outline: none;
	}

	.citation-count-badge {
		font-family: var(--font-data);
		font-size: 11px;
		color: var(--color-primary);
		background: var(--color-primary-faint);
		padding: 1px 6px;
		border-radius: var(--radius-sm);
		white-space: nowrap;
	}

	.copy-msg-btn {
		font-size: 12px;
		padding: 2px 8px;
		margin-top: var(--space-xs);
	}

	.streaming-stats {
		font-family: var(--font-data);
		font-size: 12px;
		color: var(--color-ash);
		margin-top: var(--space-xs);
	}

	.copy-tooltip {
		position: fixed;
		background: var(--color-ink);
		color: var(--color-parchment);
		font-size: 12px;
		padding: 3px 8px;
		border-radius: var(--radius-sm);
		z-index: 200;
		pointer-events: none;
		transform: translateX(-50%);
	}
</style>
