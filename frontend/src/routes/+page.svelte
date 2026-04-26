<script lang="ts">
	import { onMount } from 'svelte';
	import { corpora, activeCorpus, conversations, activeConversation, messages, streamingContent, isStreaming, theme } from '$lib/stores';
	import { listCorpora, listConversations, getConversation, createConversation, streamMessage, createCorpus, uploadDocuments, streamIngestionStatus, getCorpus, deleteDocument, reingestDocument, renameConversation, deleteConversation } from '$lib/api';
	import type { Corpus, Conversation, Document, Mode, CitationStyle, IngestionStatus, Message, CitationVerification } from '$lib/types';
	import PdfViewer from '../components/PdfViewer.svelte';
	import ConversationList from '../components/ConversationList.svelte';
	import MessageThread from '../components/MessageThread.svelte';
	import CorpusModal from '../components/CorpusModal.svelte';
	import DocumentList from '../components/DocumentList.svelte';

	let queryInput = $state('');
	let selectedMode: Mode = $state('general');
	let selectedCitationStyle: CitationStyle = $state('chicago');
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

	// Retry failed messages
	let failedMessageId = $state<string | null>(null);

	// Bulk delete conversations
	let selectMode = $state(false);
	let selectedIds = $state<Set<string>>(new Set());

	// Undo delete
	let pendingDelete = $state<{ conv: Conversation; timeout: ReturnType<typeof setTimeout> } | null>(null);

	// Mobile sidebar toggle
	let mobileMenuOpen = $state(false);

	// Citation verification results
	let verificationResults = $state<CitationVerification[]>([]);

	// Track the most recent assistant message so inline verification badges
	// only decorate citations in that turn (older turns' verifications are stale).
	let lastAssistantMsgId = $derived(
		$messages.filter(m => m.role === 'assistant').at(-1)?.id ?? null
	);

	// Toast notifications
	let toasts = $state<{ id: string; message: string; type: 'error' | 'success' | 'info' }[]>([]);
	function addToast(message: string, type: 'error' | 'success' | 'info' = 'error') {
		const id = crypto.randomUUID();
		toasts = [...toasts, { id, message, type }];
		setTimeout(() => { toasts = toasts.filter(t => t.id !== id); }, 5000);
	}
	function dismissToast(id: string) { toasts = toasts.filter(t => t.id !== id); }

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
			addToast('Document deleted', 'success');
		} catch (err) {
			docDeleteError = err instanceof Error ? err.message : 'Delete failed';
			addToast(docDeleteError!, 'error');
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
			addToast('Re-ingestion started', 'success');
		} catch (err) {
			docReingestError = err instanceof Error ? err.message : 'Re-ingest failed';
			addToast(docReingestError!, 'error');
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
		// Abort any in-progress stream
		if ($isStreaming && abortController) {
			abortController.abort();
			isStreaming.set(false);
		}
		activeCorpus.set(corpus);
		const convs = await listConversations(corpus.id);
		conversations.set(convs);
		activeConversation.set(null);
		messages.set([]);
		// Clear stale state from previous corpus
		streamingContent.set('');
		verificationResults = [];
		failedMessageId = null;
		corpusDocuments = [];
		showDocList = false;
	}

	async function selectConversation(conv: Conversation) {
		// Abort any in-progress stream
		if ($isStreaming && abortController) {
			abortController.abort();
			isStreaming.set(false);
			streamingContent.set('');
		}
		activeConversation.set(conv);
		selectedMode = conv.mode;
		verificationResults = [];
		failedMessageId = null;
		mobileMenuOpen = false;
		const detail = await getConversation(conv.id);
		messages.set(detail.messages);
	}

	async function newConversation() {
		const corpus = $activeCorpus;
		if (!corpus) return;
		const conv = await createConversation(corpus.id, selectedMode, selectedCitationStyle);
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
			verificationResults = [];
			for await (const event of streamMessage(conv.id, content, undefined, abortController.signal)) {
				switch (event.type) {
					case 'content':
						streamingContent.update(s => s + event.data);
						break;
					case 'warning':
						addToast(event.data, 'info');
						break;
					case 'verification':
						verificationResults = event.data;
						break;
					case 'error':
						addToast(event.data, 'error');
						break;
					case 'done':
						break;
				}
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
			verificationResults = [];
			for await (const event of streamMessage(conv.id, content, undefined, abortController.signal)) {
				switch (event.type) {
					case 'content':
						streamingContent.update(s => s + event.data);
						break;
					case 'warning':
						addToast(event.data, 'info');
						break;
					case 'verification':
						verificationResults = event.data;
						break;
					case 'error':
						addToast(event.data, 'error');
						break;
					case 'done':
						break;
				}
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
		const results = await Promise.allSettled(ids.map(id => deleteConversation(id)));
		const failed = results.filter(r => r.status === 'rejected').length;
		if (failed > 0) {
			addToast(`${failed} of ${ids.length} deletions failed`, 'error');
		}
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
			verificationResults = [];
			failedMessageId = null;
		}
		const timeout = setTimeout(async () => {
			pendingDelete = null;
			try {
				await deleteConversation(conv.id);
			} catch {
				addToast('Failed to delete conversation', 'error');
			}
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
		theme.update(t => {
			const next = t === 'light' ? 'dark' : 'light';
			try { localStorage.setItem('gutenberg-theme', next); } catch {}
			return next;
		});
	}

	// Restore theme from localStorage on mount
	$effect(() => {
		try {
			const saved = localStorage.getItem('gutenberg-theme');
			if (saved === 'dark' || saved === 'light') theme.set(saved);
		} catch {}
	});

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
			try {
				await renameConversation(conv.id, title);
				conversations.update(convs =>
					convs.map(c => c.id === conv.id ? { ...c, title } : c)
				);
				if ($activeConversation?.id === conv.id) {
					activeConversation.update(c => c ? { ...c, title } : c);
				}
			} catch {
				addToast('Failed to rename conversation', 'error');
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

	function showCopyTooltip(text: string, x: number, y: number) {
		navigator.clipboard.writeText(text);
		copyTooltip = { text: 'Copied!', x, y };
		setTimeout(() => { copyTooltip = null; }, 1500);
	}

	const modeLabels: Record<Mode, string> = {
		exact: 'Exact Citation',
		general: 'General',
		exhaustive: 'Exhaustive',
		precis: 'Precis',
	};

	const citationStyleLabels: Record<CitationStyle, string> = {
		chicago: 'Chicago',
		mla: 'MLA',
		apa: 'APA',
		harvard: 'Harvard',
		asa: 'ASA',
		sage: 'SAGE',
	};

	const modePlaceholders: Record<Mode, string> = {
		exact: 'Find the exact passage: "..."',
		general: 'Ask a question about the corpus...',
		exhaustive: "Find every instance of 'concept' across all works...",
		precis: 'Trace the evolution of concept X from Work A to Work B...',
	};

	function resetModal() {
		corpusName = '';
		corpusTags = '';
		selectedFiles = [];
		isDragging = false;
		ingestProgress = null;
	}

	function closeCorpusModal() {
		resetModal();
		showCorpusModal = false;
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
		<button class="btn-ghost mobile-menu-btn" onclick={() => mobileMenuOpen = !mobileMenuOpen} aria-label="Toggle conversations sidebar">
			{mobileMenuOpen ? '\u2715' : '\u2630'}
		</button>
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
		<button class="btn-ghost" onclick={toggleTheme} aria-label="Toggle dark mode">
			{$theme === 'light' ? '◐' : '◑'}
		</button>
	</div>

	<!-- Left Pane: Conversations -->
	<ConversationList
		conversations={$conversations}
		activeConversation={$activeConversation}
		isLoadingCorpora={isLoadingCorpora}
		citationCount={citationCount}
		search={conversationSearch}
		onSearchChange={(v) => conversationSearch = v}
		selectMode={selectMode}
		selectedIds={selectedIds}
		onToggleSelectMode={toggleSelectMode}
		onToggleConvSelection={toggleConvSelection}
		onDeleteSelected={deleteSelected}
		editingConvId={editingConvId}
		editTitle={editTitle}
		onEditTitleChange={(v) => editTitle = v}
		onStartRename={startRename}
		onFinishRename={finishRename}
		onCancelRename={cancelRename}
		pendingUndoLabel={pendingDelete ? 'Deleted.' : null}
		onUndoDelete={undoDelete}
		onNewConversation={newConversation}
		onSelectConversation={selectConversation}
		onDeleteConversation={deleteConversationWithUndo}
		mobileOpen={mobileMenuOpen}
	/>

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
					<DocumentList
						documents={corpusDocuments}
						isLoading={isLoadingDocuments}
						reingestingDocId={reingestingDocId}
						deletingDocId={deletingDocId}
						deleteError={docDeleteError}
						reingestError={docReingestError}
						onReingest={handleReingestDocument}
						onDelete={handleDeleteDocument}
					/>
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
			<MessageThread
				messages={$messages}
				streamingContent={$streamingContent}
				isStreaming={$isStreaming}
				verificationResults={verificationResults}
				lastAssistantMsgId={lastAssistantMsgId}
				streamStartTime={streamStartTime}
				streamElapsed={streamElapsed}
				streamingWordCount={streamingWordCount}
				bookmarksCount={bookmarks.length}
				activeConversation={$activeConversation}
				failedMessageId={failedMessageId}
				onPrint={printConversation}
				onRetry={retryFailedMessage}
				onCopyMessage={copyToClipboard}
				onOpenPdf={openPdfViewer}
				onAddBookmark={addBookmark}
				onCopyCitation={showCopyTooltip}
			/>
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
			<span class="section-label" style="margin-top: var(--space-sm); display: block;">Citation Style</span>
			<select bind:value={selectedCitationStyle} disabled={$activeConversation !== null} style="width: 100%; margin-top: var(--space-xs);" aria-label="Citation style">
				{#each (['chicago', 'mla', 'apa', 'harvard', 'asa', 'sage'] as const) as style}
					<option value={style}>{citationStyleLabels[style]}</option>
				{/each}
			</select>
			{#if $activeConversation}
				<p class="tip" style="margin-top: var(--space-xs); color: var(--color-ash); font-size: 11px;">Locked for active conversation.</p>
			{/if}
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

{#if toasts.length > 0}
	<div class="toast-container" role="region" aria-live="polite" aria-label="Notifications">
		{#each toasts as toast}
			<div class="toast toast-{toast.type}" role={toast.type === 'error' ? 'alert' : 'status'}>
				<span>{toast.message}</span>
				<button class="toast-dismiss" onclick={() => dismissToast(toast.id)} aria-label="Dismiss">&times;</button>
			</div>
		{/each}
	</div>
{/if}

{#if showCorpusModal}
	<CorpusModal
		name={corpusName}
		tags={corpusTags}
		files={selectedFiles}
		isDragging={isDragging}
		progress={ingestProgress}
		onNameChange={(v) => corpusName = v}
		onTagsChange={(v) => corpusTags = v}
		onFilesChange={(files) => selectedFiles = files}
		onDraggingChange={(v) => isDragging = v}
		onCancel={closeCorpusModal}
		onBuild={buildCorpus}
	/>
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
		font-size: 3.5rem;
		color: var(--color-primary);
		margin-bottom: var(--space-sm);
	}

	/* Conversation/Message/Document/Modal styles live in their own components.
	 * Shared utilities (loading-text, spinner, citation-*) are global in app.css. */

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

	/* Toast notifications */
	.toast-container {
		position: fixed;
		bottom: var(--space-lg);
		right: var(--space-lg);
		z-index: 300;
		display: flex;
		flex-direction: column;
		gap: var(--space-sm);
	}
	.toast {
		display: flex;
		align-items: center;
		gap: var(--space-sm);
		padding: var(--space-sm) var(--space-md);
		border-radius: var(--radius-md);
		font-size: 14px;
		animation: toast-in 0.2s ease-out;
	}
	.toast-error { background: var(--color-surface); color: var(--color-error); border: 1px solid var(--color-error); }
	.toast-success { background: var(--color-surface); color: var(--color-success); border: 1px solid var(--color-success); }
	.toast-info { background: var(--color-surface); color: var(--color-info); border: 1px solid var(--color-info); }
	.toast-dismiss {
		background: none; border: none; cursor: pointer; font-size: 16px;
		color: inherit; opacity: 0.6; padding: 0 2px;
	}
	.toast-dismiss:hover { opacity: 1; }
	@keyframes toast-in { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

	/* verification-*, citation-badge, citation-verify: see app.css (global) and
	 * MessageThread.svelte (summary-panel scoped styles). */
</style>
