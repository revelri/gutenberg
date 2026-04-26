<script lang="ts">
	let { source, page, quote, onclose }: {
		source: string;
		page: number;
		quote: string;
		onclose: () => void;
	} = $props();

	let currentPage = $state(page);
	let zoomLevel = $state(100);
	let iframeEl: HTMLIFrameElement | undefined = $state(undefined);
	let loading = $state(true);
	let error = $state(false);

	const searchText = quote.slice(0, 120);

	function buildViewerUrl(pg: number): string {
		return `/pdfjs/web/viewer.html?file=/api/pdf/${encodeURIComponent(source)}#page=${pg}&search=${encodeURIComponent(searchText)}`;
	}

	let viewerUrl = $state(buildViewerUrl(currentPage));

	function goToPage(pg: number) {
		if (pg < 1) return;
		currentPage = pg;
		loading = true;
		error = false;
		viewerUrl = buildViewerUrl(pg);
	}

	function prevPage() { goToPage(currentPage - 1); }
	function nextPage() { goToPage(currentPage + 1); }

	function zoomIn() { zoomLevel = Math.min(zoomLevel + 25, 300); }
	function zoomOut() { zoomLevel = Math.max(zoomLevel - 25, 25); }
	function resetZoom() { zoomLevel = 100; }

	function onIframeLoad() { loading = false; }
	function onIframeError() { loading = false; error = true; }
</script>

<svelte:window onkeydown={(e) => e.key === 'Escape' && onclose()} />

<div class="pdf-modal-overlay" role="dialog" aria-modal="true" aria-label="PDF Viewer: {source}" onclick={onclose}>
	<div class="pdf-modal" onclick={(e) => e.stopPropagation()}>
		<div class="pdf-header">
			<div class="pdf-nav-controls">
				<button class="btn-ghost pdf-btn" onclick={prevPage} disabled={currentPage <= 1} aria-label="Previous page">&#8249;</button>
				<span class="pdf-page-indicator">p. {currentPage}</span>
				<button class="btn-ghost pdf-btn" onclick={nextPage} aria-label="Next page">&#8250;</button>
			</div>
			<span class="pdf-title">{source}</span>
			<div class="pdf-zoom-controls">
				<button class="btn-ghost pdf-btn" onclick={zoomOut} disabled={zoomLevel <= 25} aria-label="Zoom out">&minus;</button>
				<span class="pdf-zoom-indicator">{zoomLevel}%</span>
				<button class="btn-ghost pdf-btn" onclick={zoomIn} disabled={zoomLevel >= 300} aria-label="Zoom in">+</button>
				<button class="btn-ghost pdf-btn" onclick={resetZoom} aria-label="Reset zoom">Reset</button>
			</div>
			<button class="btn-ghost" onclick={onclose} aria-label="Close PDF viewer">&#x2715;</button>
		</div>
		<div class="pdf-iframe-wrapper">
			{#if loading}
				<div class="pdf-loading" role="status">Loading PDF...</div>
			{/if}
			{#if error}
				<div class="pdf-error">PDF could not be loaded. The file may not be available.</div>
			{:else}
				<iframe
					bind:this={iframeEl}
					src={viewerUrl}
					title="PDF viewer for {source}"
					style="transform: scale({zoomLevel / 100}); transform-origin: top left;"
					onload={onIframeLoad}
					onerror={onIframeError}
				></iframe>
			{/if}
		</div>
	</div>
</div>

<style>
	.pdf-modal-overlay {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.6);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 200;
	}

	.pdf-modal {
		width: 90vw;
		height: 90vh;
		background: var(--color-surface);
		border-radius: var(--radius-lg);
		overflow: hidden;
		display: flex;
		flex-direction: column;
	}

	.pdf-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: var(--space-sm) var(--space-md);
		border-bottom: 1px solid var(--color-parchment);
		background: var(--color-surface);
		gap: var(--space-md);
	}

	.pdf-title {
		font-size: 14px;
		color: var(--color-stone);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		flex: 1;
	}

	.pdf-nav-controls,
	.pdf-zoom-controls {
		display: flex;
		align-items: center;
		gap: var(--space-xs);
		flex-shrink: 0;
	}

	.pdf-btn {
		padding: var(--space-xs) var(--space-sm);
		text-decoration: none;
		font-size: 16px;
		line-height: 1;
		min-width: 28px;
		text-align: center;
	}

	.pdf-page-indicator,
	.pdf-zoom-indicator {
		font-size: 12px;
		color: var(--color-ash);
		font-family: var(--font-data);
		min-width: 32px;
		text-align: center;
	}

	.pdf-iframe-wrapper {
		flex: 1;
		overflow: auto;
		position: relative;
	}

	.pdf-iframe-wrapper iframe {
		border: none;
		width: 100%;
		height: 100%;
	}

	.pdf-loading, .pdf-error {
		display: flex;
		align-items: center;
		justify-content: center;
		height: 100%;
		font-family: var(--font-body);
		font-size: 14px;
		color: var(--color-stone);
	}

	.pdf-error {
		color: var(--color-error);
	}
</style>
