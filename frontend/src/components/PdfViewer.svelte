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

	const MAX_PAGE = 20;

	const searchText = quote.slice(0, 60);

	function buildViewerUrl(pg: number): string {
		return `/pdfjs/web/viewer.html?file=/api/pdf/${encodeURIComponent(source)}#page=${pg}&search=${encodeURIComponent(searchText)}`;
	}

	let viewerUrl = $state(buildViewerUrl(currentPage));

	function goToPage(pg: number) {
		if (pg < 1 || pg > MAX_PAGE) return;
		currentPage = pg;
		viewerUrl = buildViewerUrl(pg);
	}

	function prevPage() { goToPage(currentPage - 1); }
	function nextPage() { goToPage(currentPage + 1); }

	function zoomIn() {
		zoomLevel = Math.min(zoomLevel + 25, 300);
	}

	function zoomOut() {
		zoomLevel = Math.max(zoomLevel - 25, 25);
	}

	function resetZoom() {
		zoomLevel = 100;
	}

	let iframeTransform = $derived(`scale(${zoomLevel / 100})`);
</script>

<svelte:window onkeydown={(e) => e.key === 'Escape' && onclose()} />

<div class="pdf-modal-overlay" role="dialog" onclick={onclose}>
	<div class="pdf-modal" onclick={(e) => e.stopPropagation()}>
		<div class="pdf-header">
			<div class="pdf-nav-controls">
				<button class="btn-ghost pdf-btn" onclick={prevPage} disabled={currentPage <= 1}>&#8249;</button>
				<span class="pdf-page-indicator">p. {currentPage}</span>
				<button class="btn-ghost pdf-btn" onclick={nextPage} disabled={currentPage >= MAX_PAGE}>&#8250;</button>
			</div>
			<span class="pdf-title">{source}</span>
			<div class="pdf-zoom-controls">
				<button class="btn-ghost pdf-btn" onclick={zoomOut} disabled={zoomLevel <= 25}>&minus;</button>
				<span class="pdf-zoom-indicator">{zoomLevel}%</span>
				<button class="btn-ghost pdf-btn" onclick={zoomIn} disabled={zoomLevel >= 300}>+</button>
				<button class="btn-ghost pdf-btn" onclick={resetZoom}>Reset</button>
			</div>
			<button class="btn-ghost" onclick={onclose}>&#x2715;</button>
		</div>
		<div class="pdf-iframe-wrapper">
			<iframe bind:this={iframeEl} src={viewerUrl} title="PDF viewer" style="transform: {iframeTransform}; transform-origin: top left;"></iframe>
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
		padding: 2px 8px;
		text-decoration: none;
		font-size: 16px;
		line-height: 1;
		min-width: 28px;
		text-align: center;
	}

	.pdf-btn:disabled {
		opacity: 0.3;
		cursor: default;
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
		width: calc(100% / (var(--zoom-factor, 1)));
		height: calc(100% / (var(--zoom-factor, 1)));
		min-width: 100%;
		min-height: 100%;
	}

	iframe {
		border: none;
		width: 100%;
	}
</style>
