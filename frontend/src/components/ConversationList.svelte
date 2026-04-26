<script lang="ts">
	import type { Conversation } from '$lib/types';

	interface Props {
		conversations: Conversation[];
		activeConversation: Conversation | null;
		isLoadingCorpora: boolean;
		citationCount: number;

		search: string;
		onSearchChange: (v: string) => void;

		selectMode: boolean;
		selectedIds: Set<string>;
		onToggleSelectMode: () => void;
		onToggleConvSelection: (id: string) => void;
		onDeleteSelected: () => void;

		editingConvId: string | null;
		editTitle: string;
		onEditTitleChange: (v: string) => void;
		onStartRename: (conv: Conversation) => void;
		onFinishRename: (conv: Conversation) => void;
		onCancelRename: () => void;

		pendingUndoLabel: string | null;
		onUndoDelete: () => void;

		onNewConversation: () => void;
		onSelectConversation: (conv: Conversation) => void;
		onDeleteConversation: (conv: Conversation) => void;

		mobileOpen: boolean;
	}

	let {
		conversations,
		activeConversation,
		isLoadingCorpora,
		citationCount,
		search,
		onSearchChange,
		selectMode,
		selectedIds,
		onToggleSelectMode,
		onToggleConvSelection,
		onDeleteSelected,
		editingConvId,
		editTitle,
		onEditTitleChange,
		onStartRename,
		onFinishRename,
		onCancelRename,
		pendingUndoLabel,
		onUndoDelete,
		onNewConversation,
		onSelectConversation,
		onDeleteConversation,
		mobileOpen,
	}: Props = $props();

	let filtered = $derived(
		conversations.filter(
			c => !search || (c.title || '').toLowerCase().includes(search.toLowerCase())
		)
	);
</script>

<div class="left-pane" class:mobile-open={mobileOpen}>
	<div class="pane-header">
		<span class="section-label header-label">Conversations</span>
		{#if conversations.length > 0}
			<button class="btn-ghost select-toggle-btn" onclick={onToggleSelectMode}>
				{selectMode ? 'Cancel' : 'Select'}
			</button>
		{/if}
		<button class="btn-primary new-btn" onclick={onNewConversation}>+</button>
	</div>

	{#if conversations.length > 0}
		<div class="search-wrap">
			<input
				type="text"
				placeholder="Search conversations..."
				value={search}
				oninput={(e) => onSearchChange((e.target as HTMLInputElement).value)}
				class="search-input"
			/>
		</div>
	{/if}

	<div class="conv-list">
		{#if conversations.length === 0 && isLoadingCorpora}
			<div class="loading-text">
				<span class="spinner"></span>
				Loading...
			</div>
		{:else}
			{#each filtered as conv}
				<div class="conv-row">
					{#if selectMode}
						<input
							type="checkbox"
							checked={selectedIds.has(conv.id)}
							onchange={() => onToggleConvSelection(conv.id)}
							class="conv-checkbox"
							aria-label="Select conversation"
						/>
					{/if}
					<button
						class="conv-item"
						class:active={activeConversation?.id === conv.id}
						onclick={() => {
							if (!selectMode && editingConvId !== conv.id) onSelectConversation(conv);
						}}
					>
						{#if editingConvId === conv.id}
							<input
								type="text"
								value={editTitle}
								oninput={(e) => onEditTitleChange((e.target as HTMLInputElement).value)}
								class="rename-input"
								onclick={(e) => e.stopPropagation()}
								onkeydown={(e) => {
									if (e.key === 'Enter') {
										e.preventDefault();
										onFinishRename(conv);
									} else if (e.key === 'Escape') {
										e.preventDefault();
										onCancelRename();
									}
									e.stopPropagation();
								}}
								onblur={() => onFinishRename(conv)}
							/>
						{:else}
							<span
								class="conv-title"
								onclick={(e) => {
									e.stopPropagation();
									if (!selectMode) onStartRename(conv);
								}}
								role="presentation"
							>{conv.title || 'New conversation'}</span>
							<span class="mode-badge">{conv.mode}</span>
							{#if activeConversation?.id === conv.id && citationCount > 0}
								<span class="citation-count-badge">{citationCount}</span>
							{/if}
						{/if}
					</button>
					{#if !selectMode}
						<button
							class="btn-ghost conv-delete-btn"
							onclick={() => onDeleteConversation(conv)}
							aria-label="Delete conversation"
						>&times;</button>
					{/if}
				</div>
			{/each}

			{#if selectMode && selectedIds.size > 0}
				<button class="btn-primary delete-selected-btn" onclick={onDeleteSelected}>
					Delete Selected ({selectedIds.size})
				</button>
			{/if}

			{#if pendingUndoLabel}
				<div class="undo-toast">
					<span>{pendingUndoLabel}</span>
					<button class="btn-ghost" onclick={onUndoDelete}>Undo?</button>
				</div>
			{/if}

			{#if conversations.length === 0}
				<p class="empty">
					No conversations yet.<br>Click + to start.
				</p>
			{/if}
		{/if}
	</div>
</div>

<style>
	.pane-header {
		padding: var(--space-md);
		display: flex;
		align-items: center;
		gap: var(--space-sm);
	}
	.header-label { flex: 1; }
	.new-btn { padding: 4px 12px; font-size: 13px; }

	.search-wrap { padding: 0 var(--space-sm) var(--space-sm); }
	.search-input {
		width: 100%;
		padding: 6px 10px;
		font-size: 13px;
		border: 1px solid var(--color-parchment);
		border-radius: var(--radius-sm);
		background: transparent;
		font-family: var(--font-body);
		color: var(--color-ink);
	}

	.conv-list { padding: 0 var(--space-sm); }

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
	.conv-item.active {
		background: var(--color-primary-faint);
		border-left: 2px solid var(--color-primary);
	}
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

	.empty {
		color: var(--color-ash);
		font-size: 13px;
		padding: var(--space-md);
		text-align: center;
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
</style>
