<script lang="ts">
	import type { IngestionStatus } from '$lib/types';

	interface Props {
		name: string;
		tags: string;
		files: File[];
		isDragging: boolean;
		progress: IngestionStatus | null;
		onNameChange: (v: string) => void;
		onTagsChange: (v: string) => void;
		onFilesChange: (files: File[]) => void;
		onDraggingChange: (v: boolean) => void;
		onCancel: () => void;
		onBuild: () => void;
	}

	let {
		name,
		tags,
		files,
		isDragging,
		progress,
		onNameChange,
		onTagsChange,
		onFilesChange,
		onDraggingChange,
		onCancel,
		onBuild,
	}: Props = $props();

	let fileInputEl: HTMLInputElement | undefined = $state(undefined);

	function formatFileSize(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
	}

	function handleFileSelect(e: Event) {
		const input = e.target as HTMLInputElement;
		if (!input.files) return;
		const next = [...files];
		for (const f of input.files) {
			if (!next.find(x => x.name === f.name)) next.push(f);
		}
		onFilesChange(next);
		input.value = '';
	}

	function handleDrop(e: DragEvent) {
		e.preventDefault();
		onDraggingChange(false);
		if (!e.dataTransfer?.files) return;
		const next = [...files];
		for (const f of e.dataTransfer.files) {
			if (!next.find(x => x.name === f.name)) next.push(f);
		}
		onFilesChange(next);
	}

	function handleDragOver(e: DragEvent) {
		e.preventDefault();
		onDraggingChange(true);
	}

	function handleDragLeave() {
		onDraggingChange(false);
	}

	function removeFile(index: number) {
		onFilesChange(files.filter((_, i) => i !== index));
	}
</script>

<div
	class="modal-overlay"
	role="dialog"
	aria-modal="true"
	aria-label="New Corpus"
	onclick={onCancel}
>
	<div class="modal-content card" onclick={(e) => e.stopPropagation()} role="presentation">
		<h3>New Corpus</h3>
		<p class="modal-desc">Create a corpus project and upload documents for ingestion.</p>

		<label class="field-label">
			Name
			<input
				type="text"
				value={name}
				oninput={(e) => onNameChange((e.target as HTMLInputElement).value)}
				placeholder="e.g. Deleuze Critical Theory"
			/>
		</label>

		<label class="field-label field-label-spaced">
			Tags
			<input
				type="text"
				value={tags}
				oninput={(e) => onTagsChange((e.target as HTMLInputElement).value)}
				placeholder="philosophy, deleuze, poststructuralism"
			/>
		</label>

		<div
			class="drop-zone"
			class:dragging={isDragging}
			ondrop={handleDrop}
			ondragover={handleDragOver}
			ondragleave={handleDragLeave}
			role="presentation"
		>
			<p>Drag & drop PDF, EPUB, or DOCX files here</p>
			<p class="drop-or">or</p>
			<button class="btn-secondary" onclick={() => fileInputEl?.click()}>Browse Files</button>
			<input
				bind:this={fileInputEl}
				type="file"
				multiple
				accept=".pdf,.epub,.docx,.txt,.md"
				onchange={handleFileSelect}
				class="file-input"
			/>
		</div>

		{#if files.length > 0}
			<div class="file-list">
				{#each files as file, i}
					<div class="file-item">
						<span class="file-name">{file.name}</span>
						<span class="file-size">{formatFileSize(file.size)}</span>
						<button onclick={() => removeFile(i)} aria-label="Remove file">&times;</button>
					</div>
				{/each}
			</div>
		{/if}

		{#if progress}
			<div class="progress-bar-container">
				<div class="progress-bar">
					<div
						class="progress-bar-fill"
						style="width: {progress.total_files > 0
							? Math.round((progress.completed_files / progress.total_files) * 100)
							: 0}%"
					></div>
				</div>
				<p class="progress-text">
					{#if progress.status === 'failed'}
						Failed: {progress.error || 'Unknown error'}
					{:else if progress.status === 'done'}
						Complete — {progress.completed_files}/{progress.total_files} files processed
					{:else}
						{progress.completed_files}/{progress.total_files} files
						{#if progress.current_file}
							&mdash; processing {progress.current_file}
						{/if}
					{/if}
				</p>
			</div>
		{/if}

		<div class="modal-actions">
			<button class="btn-secondary" onclick={onCancel}>Cancel</button>
			<button
				class="btn-primary"
				onclick={onBuild}
				disabled={!name.trim() || files.length === 0 || progress?.status === 'running'}
			>
				{progress?.status === 'running' ? 'Building...' : 'Build'}
			</button>
		</div>
	</div>
</div>

<style>
	.modal-overlay {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.4);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 100;
	}
	.modal-content { width: 520px; max-height: 80vh; overflow-y: auto; padding: var(--space-lg); }

	.modal-desc { color: var(--color-stone); margin: var(--space-sm) 0 var(--space-md); }

	.field-label { font-size: 14px; font-weight: 600; }
	.field-label input { width: 100%; margin-top: 4px; }
	.field-label-spaced { margin-top: var(--space-md); display: block; }

	.drop-zone {
		margin-top: var(--space-md);
		border: 2px dashed var(--color-parchment);
		border-radius: var(--radius-md);
		padding: var(--space-xl);
		text-align: center;
		color: var(--color-stone);
		transition: border-color var(--duration-short) ease-out;
	}
	.drop-zone:hover { border-color: var(--color-primary); }
	.drop-zone.dragging {
		border-color: var(--color-primary) !important;
		background: var(--color-primary-faint);
	}
	.drop-or { color: var(--color-ash); font-size: 13px; margin: var(--space-xs) 0; }
	.file-input { display: none; }

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
	.file-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
	.file-size { color: var(--color-ash); font-size: 12px; white-space: nowrap; }

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

	.modal-actions {
		display: flex;
		gap: var(--space-sm);
		margin-top: var(--space-lg);
		justify-content: flex-end;
	}
</style>
