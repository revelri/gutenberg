import type { CitationVerification } from './types';

export function escAttr(s: string): string {
	return s
		.replace(/&/g, '&amp;')
		.replace(/"/g, '&quot;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;');
}

export function formatResponse(text: string, verifications: CitationVerification[] = []): string {
	const vMap = new Map<string, CitationVerification>();
	for (const v of verifications) {
		if (!v.source || v.status === 'too_short') continue;
		const src = v.source.toLowerCase();
		vMap.set(`${src}|${v.page ?? ''}`, v);
		if (!vMap.has(`${src}|`)) vMap.set(`${src}|`, v);
	}

	const verifyMarker = (source: string, page?: string): string => {
		const key = `${source.toLowerCase()}|${page ?? ''}`;
		const v = vMap.get(key) ?? vMap.get(`${source.toLowerCase()}|`);
		if (!v) return '';
		const icon = v.status === 'verified' ? '\u2713' : v.status === 'approximate' ? '\u2248' : '\u2717';
		const pct = (v.similarity * 100).toFixed(0);
		const title = v.status === 'verified'
			? `Verified against source (${pct}% match)`
			: v.status === 'approximate'
				? `Approximate match (${pct}%) — wording differs slightly`
				: 'Could not verify against source';
		return ` <span class="citation-verify ${v.status}" title="${escAttr(title)}" aria-label="${escAttr(title)}">${icon}</span>`;
	};

	return text
		.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
		.replace(/\n/g, '<br>')
		.replace(/&lt;blockquote&gt;/g, '<blockquote>').replace(/&lt;\/blockquote&gt;/g, '</blockquote>')
		.replace(/&quot;([^&]{20,}?)&quot;/g, '<blockquote>&quot;$1&quot;</blockquote>')
		.replace(/"([^"]{20,}?)"/g, '<blockquote>"$1"</blockquote>')
		.replace(
			/\[Source:\s*([^,\]]+),\s*p\.\s*(\d+)\]/g,
			(_m, source, page) => {
				const safeSrc = escAttr(source);
				return `<span class="citation-badge" data-action="copy" data-source="${safeSrc}" data-page="${page}">[Source: ${source}, p. ${page}]</span>`
					+ verifyMarker(source, page)
					+ ` <button class="btn-ghost view-pdf-btn" data-action="view" data-source="${safeSrc}" data-page="${page}" aria-label="View in PDF">View</button>`
					+ ` <button class="btn-ghost bookmark-btn" data-action="bookmark" data-source="${safeSrc}" data-page="${page}" aria-label="Bookmark citation">&#9734;</button>`;
			}
		)
		.replace(
			/\[Source:\s*([^\]]+)\]/g,
			(_m, source) => {
				const safeSrc = escAttr(source);
				return `<span class="citation-badge" data-action="copy" data-source="${safeSrc}">[Source: ${source}]</span>`
					+ verifyMarker(source);
			}
		);
}
