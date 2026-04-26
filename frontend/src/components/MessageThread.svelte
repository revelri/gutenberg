<script lang="ts">
	import type { CitationVerification, Conversation, Message } from '$lib/types';
	import { formatResponse } from '$lib/citations';

	interface Props {
		messages: Message[];
		streamingContent: string;
		isStreaming: boolean;
		verificationResults: CitationVerification[];
		lastAssistantMsgId: string | null;
		streamStartTime: number | null;
		streamElapsed: number;
		streamingWordCount: number;
		bookmarksCount: number;
		activeConversation: Conversation | null;
		failedMessageId: string | null;
		onPrint: () => void;
		onRetry: (msgId: string) => void;
		onCopyMessage: (text: string) => void;
		onOpenPdf: (source: string, page: number, quote: string) => void;
		onAddBookmark: (quote: string, source: string, page: number) => void;
		onCopyCitation: (text: string, x: number, y: number) => void;
	}

	let {
		messages,
		streamingContent,
		isStreaming,
		verificationResults,
		lastAssistantMsgId,
		streamStartTime,
		streamElapsed,
		streamingWordCount,
		bookmarksCount,
		activeConversation,
		failedMessageId,
		onPrint,
		onRetry,
		onCopyMessage,
		onOpenPdf,
		onAddBookmark,
		onCopyCitation,
	}: Props = $props();

	function handleResponseClick(e: MouseEvent) {
		const el = (e.target as HTMLElement).closest('[data-action]') as HTMLElement | null;
		if (!el) return;
		const action = el.dataset.action;
		const source = el.dataset.source ?? '';
		const page = el.dataset.page ? Number(el.dataset.page) : 0;
		if (action === 'copy') {
			onCopyCitation(el.textContent ?? '', e.clientX, e.clientY);
		} else if (action === 'view') {
			const quote = el.closest('.response-content')?.querySelector('blockquote')?.textContent?.slice(0, 120) ?? '';
			onOpenPdf(source, page, quote);
		} else if (action === 'bookmark') {
			const quote = el.closest('.response-content')?.querySelector('blockquote')?.textContent?.slice(0, 200) ?? '';
			onAddBookmark(quote, source, page);
		}
	}
</script>

<div class="messages-container">
	<div class="messages-toolbar">
		{#if bookmarksCount > 0}
			<span class="bookmark-count" title="{bookmarksCount} bookmarked citations">&#9734; {bookmarksCount}</span>
		{/if}
		<button
			class="btn-ghost print-btn"
			onclick={onPrint}
			disabled={!activeConversation}
		>Print</button>
	</div>

	{#each messages as msg}
		<div class="message" class:user={msg.role === 'user'} class:assistant={msg.role === 'assistant'}>
			{#if msg.role === 'user'}
				<div class="message-label section-label">Query</div>
				<p class="query-text">{msg.content}</p>
				{#if failedMessageId === msg.id}
					<button class="btn-ghost retry-btn" onclick={() => onRetry(msg.id)}>Retry</button>
				{/if}
			{:else}
				<div class="message-label section-label">Response</div>
				<div
					class="response-content"
					onclick={handleResponseClick}
					role="presentation"
				>{@html formatResponse(msg.content, msg.id === lastAssistantMsgId ? verificationResults : [])}</div>
				<button class="btn-ghost copy-msg-btn" onclick={() => onCopyMessage(msg.content)}>Copy</button>
			{/if}
		</div>
		<hr />
	{/each}

	{#if streamingContent}
		<div class="message assistant">
			<div class="message-label section-label">Response</div>
			<div
				class="response-content streaming"
				onclick={handleResponseClick}
				role="presentation"
			>{@html formatResponse(streamingContent, [])}</div>
			{#if streamStartTime}
				<div class="streaming-stats">
					{streamingWordCount} words &middot; {streamElapsed.toFixed(1)}s
				</div>
			{/if}
		</div>
	{/if}

	{#if verificationResults.length > 0 && !isStreaming}
		<div class="verification-summary">
			<span class="section-label">Citation Verification</span>
			<div class="verification-badges">
				{#each verificationResults as v}
					{#if v.status !== 'too_short'}
						<div
							class="verification-badge"
							class:verified={v.status === 'verified'}
							class:approximate={v.status === 'approximate'}
							class:unverified={v.status === 'unverified'}
						>
							<span class="v-icon">{v.status === 'verified' ? '\u2713' : v.status === 'approximate' ? '\u2248' : '\u2717'}</span>
							<span class="v-quote">{v.quote}</span>
							{#if v.source}
								<span class="v-source">{v.source}, p. {v.page}</span>
							{/if}
						</div>
					{/if}
				{/each}
			</div>
		</div>
	{/if}
</div>

<style>
	.messages-container { max-width: 800px; margin: 0 auto; }
	.messages-toolbar {
		display: flex;
		justify-content: flex-end;
		align-items: center;
		gap: var(--space-sm);
		padding: var(--space-xs) 0 var(--space-sm);
	}
	.bookmark-count { color: var(--color-ash); font-size: 13px; font-family: var(--font-data); }
	.print-btn { font-size: 13px; padding: 2px 8px; }

	.query-text { font-weight: 600; }
	.retry-btn {
		font-size: 12px;
		padding: 2px 8px;
		margin-top: var(--space-xs);
		text-decoration: none;
	}
	.response-content { line-height: 1.7; }

	.verification-summary {
		margin-top: var(--space-md);
		padding: var(--space-sm) var(--space-md);
		background: var(--color-vellum);
		border-radius: var(--radius-md);
	}
	.verification-badges {
		display: flex;
		flex-direction: column;
		gap: var(--space-xs);
		margin-top: var(--space-xs);
	}
	.verification-badge {
		display: flex;
		align-items: baseline;
		gap: var(--space-xs);
		font-size: 13px;
		font-family: var(--font-data);
		padding: 2px var(--space-xs);
		border-radius: var(--radius-sm);
	}
	.verification-badge.verified .v-icon { color: var(--color-success); }
	.verification-badge.approximate .v-icon { color: var(--color-warning); }
	.verification-badge.unverified .v-icon { color: var(--color-error); }
	.v-quote { color: var(--color-stone); max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
	.v-source { color: var(--color-ash); font-size: 12px; }

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
	.message { margin-bottom: var(--space-md); }
	.message-label { margin-bottom: var(--space-xs); }
	.streaming { opacity: 0.85; }
</style>
