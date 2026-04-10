<script lang="ts">
	import '../app.css';
	import { theme } from '$lib/stores';

	let { children } = $props();

	$effect(() => {
		document.documentElement.setAttribute('data-theme', $theme);
	});
</script>

<svelte:head>
	<title>Gutenberg</title>
</svelte:head>

{@render children()}
