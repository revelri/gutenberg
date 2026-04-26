<script lang="ts">
	import type { Document } from '$lib/types';

	interface Props {
		documents: Document[];
		isLoading: boolean;
		reingestingDocId: string | null;
		deletingDocId: string | null;
		deleteError: string | null;
		reingestError: string | null;
		onReingest: (docId: string) => void;
		onDelete: (docId: string) => void;
	}

	let {
		documents,
		isLoading,
		reingestingDocId,
		deletingDocId,
		deleteError,
		reingestError,
		onReingest,
		onDelete,
	}: Props = $props();
</script>

<div class="doc-list-card">
	{#if isLoading}
		<div class="loading-text"><span class="spinner"></span> Loading...</div>
	{:else if documents.length === 0}
		<p class="empty">No documents found.</p>
	{:else}
		{#each documents as doc}
			<div class="doc-item">
				<div class="doc-info">
					<span class="doc-name">{doc.filename}</span>
					<span class="doc-chunks">{doc.chunks} chunks</span>
					{#if doc.error}
						<span class="doc-error">{doc.error}</span>
					{/if}
				</div>
				<button
					class="btn-ghost doc-action-btn"
					disabled={reingestingDocId === doc.id}
					onclick={() => onReingest(doc.id)}
					title="Re-ingest"
					aria-label="Re-ingest document"
				>
					{#if reingestingDocId === doc.id}<span class="spinner spinner-sm"></span>{:else}&#8635;{/if}
				</button>
				<button
					class="btn-ghost doc-action-btn"
					disabled={deletingDocId === doc.id}
					onclick={() => onDelete(doc.id)}
					title="Delete"
					aria-label="Delete document"
				>
					{#if deletingDocId === doc.id}<span class="spinner spinner-sm"></span>{:else}&times;{/if}
				</button>
			</div>
		{/each}
		{#if deleteError}
			<p class="action-error">{deleteError}</p>
		{/if}
		{#if reingestError}
			<p class="action-error">{reingestError}</p>
		{/if}
	{/if}
</div>

<style>
	.doc-list-card {
		margin-top: var(--space-md);
		padding: var(--space-md);
		background: var(--color-vellum);
		border-radius: var(--radius-md);
		text-align: left;
		max-width: 600px;
	}
	.doc-item {
		display: flex;
		align-items: center;
		gap: var(--space-sm);
		padding: var(--space-xs) 0;
		border-bottom: 1px solid var(--color-parchment);
	}
	.doc-item:last-child { border-bottom: none; }
	.doc-info { flex: 1; text-align: left; }
	.doc-name { font-weight: 600; font-size: 13px; }
	.doc-chunks { color: var(--color-ash); font-size: 12px; margin-left: var(--space-sm); }
	.doc-error { color: var(--color-error); font-size: 12px; margin-left: var(--space-sm); }
	.doc-action-btn {
		font-size: 14px;
		padding: 4px 8px;
		text-decoration: none;
		line-height: 1;
	}
	.empty { color: var(--color-ash); font-size: 13px; }
	.action-error { color: var(--color-error); font-size: 12px; margin-top: var(--space-sm); }
	.spinner-sm { width: 12px; height: 12px; }
</style>
