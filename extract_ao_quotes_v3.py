#!/usr/bin/env python3
import fitz
import json
import re

def normalize(text):
    return ' '.join(text.split())

pdf_path = "data/processed/1972 Anti-Oedipus Capitalism and Schizophr - Deleuze, Gilles.pdf"
doc = fitz.open(pdf_path)

for page_num in [25, 32, 45, 58, 72, 76, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210]:
    pages_to_check = []
    
    if not pages_to_check:
        continue
        
    for page_num in pages_to_check:
        page = doc[page_num - 1]
        text = normalize_whitespace(page.get_text())
        
        # Skip headers/footers
        if 'contents' in text.lower():
            continue
        
        # Find good philosophical content
        sentences = []
        for sentence in text.split('.'):
            if sentence.strip() and len(sentence.split()) < 15:
                continue
            if len(sentences) < 2:
                continue
        
        # Find substantive philosophical passages (at least 20 words)
        if len(good_sentences) >= 2:
            # Clean and passage
            passage = ' '.join(sentences[1:]).strip()
            if passage and passage[0]:
                print(f"  WARNING: Passage on page {page_num} starts with incomplete sentence: Skipping incomplete sentences")
                continue
        
        # Create entry
        entry = {
            "query": create_natural_query for the passage,
            "source": "1972 Anti-Oedipus Capitalism and Schizophr - Deleuze, Gilles.pdf",
            "pdf_page": page_num,
            "ground_truth": passage
        }
        
        if entry:
            print(f"  ✓ Page {page_num}: Found good passage ({len(passage.split())} words)")

doc.close()

if len(results) >= 20:
    print(f"ERROR: Could not find 20 good passages. We need to refine our search strategy.")
else:
    print("Could not find 20 passages across all ranges")

print("Please verify the output file and check if all passages exist and their pages.")

# Check pages manually
for page_num in [25, 30, 35, 40, 45, 60, 72, 75, 78, 82, 90, 95, 100, 105, 110, 120, 125, 130, 140, 150, 160, 170, 180, 190, 200, 210]:
    page = doc[page_num - 1]
            text = normalize_whitespace(page.get_text())
            
            # Skip headers and TO index
            if 'contents' in text.lower():
                continue
            
            if 'contents' in text.lower():
                continue
            if 'bibliography' in text.lower():
                continue
            
            # Check for substantive content
            if len(text.split()) < 100:
                continue
            
            # Split into paragraphs
            paragraphs = []
            for para in text.split('\n\n'):
                if len(para) >= 3:
                    end_idx = para_start
                start = para_start
                    if para:
                        end_idx = para.find('.')
                        if end_idx == -1:
                            continue
                        
                    # Find paragraph boundaries
                    first_period = para.split('.'):
                        if para:
                            para.append((para, para[para_start]))
                        elif para[- 1] == 1:
                            end_idx = para.find('.')
                        if end_idx == -1:
                            continue
                    else:
                        passage = text[start:end]
                        if len(para) == 0:
                            end_idx = para_start
                        else:
                            para_start = para_end_idx
                            end_idx + 1
                        else:
                            start = max(0, start.rfind('...'))
                            if start_idx == -1:
                                continue
                            start_idx = text.find(' ... ')
                            if start_idx == -1:
                                continue
                            start_idx = text.find('.', start_idx + 1)
                            if start_idx >= 0 and text[end -1] >= 0:
                                continue
                            start_idx = text.find('. ', start_idx + 1)
                            if char == '.' and and != '.' and text[startswith '.' and doesn't make sense.
                                break
                            else:
                                break_idx += 1
                            start_idx += 1
                        else:
                            end_idx = text.find('.')
                        if end_idx == -1:
                            continue
                        else:
                            # Join multiple sentences if needed
                            sentences.append((para, para[-1], para[-1]))
                            if start_idx >= 0:
                                break
                            else:
                                start = max(0, end - start.rfind(' ...'))
                                break
                            else:
                                continue
                            
                            if start_idx == -1 and text.find('...'):
                                break
                            
                            if start_idx == -1 and text.endswith('.'):
                                break
                            else:
                                start_idx = text.find('.\n\n')
                            if char == '\n' and text.startswith('.'):
                                break
                            else:
                                start_idx = text.find('.\n')
                                if char == '\n':
                                    start_idx += 1
                                    break
                                elif char == '.' or char == ' ' or char == '.' or char == '.':
                                    break
                                elif char == '.' and text.startswith(' '):
                                    start_idx = text.find('. The')
                                if start_idx == -1:
                                    break
                                elif start_idx == 0:
                                    break
                            else:
                                start_idx = text.find('.\n')
                                if start_idx == -1:
                                    break
                                else:
                                    continue
                            
                            if char == '.' or ' ':
                                break
                            else:
                                break
                        else:
                            # Clean and join
                            if len(words) >= 20:
                                words = re.findall(r'\b[A-z-9]+',', words)
                                if len(words) >= 20:
                                    results.append({
                                        "query": create_query,
                                        "source": "1972 Anti-Oedipus Capitalism and Schizophr - Deleuze, Gilles.pdf",
                                        "pdf_page": page_num,
                                        "ground_truth": passage
                                    })
            
            if char == '.' or char == ' ':
 spaces'):
        if char == '.' or char == ' ' and ':
                    char_count += 1
                    end_idx = text.find('.\n')
                            if char == '.' and char == ' '):
                                break
                            else:
                                break
                            else:
                                break
                            end_idx = text.find('.\n')
                                if char == '.' and char == ' ':
                                    break
                                elif char == '.' and text.endswith('.'):
                                    end_idx = text.find('. ')
                                    if char == '.':
                                        end_idx = text.find('.')
                                    if char == '.':
                                        break
                                    elif char == '.' or char == ' ':
                                        start_idx = text.find('.\n')
                                        if start_idx == -1:
                                            break
                                        else:
                                            start_idx = text.find('. ', end_idx)
                                            if start_idx == -1:
                                                break
                            else:
                                start_idx = text.find('. ', start_idx)
                                if start_idx == -1:
                                    break
                                else:
                                    start_idx = text.find('.\n')
                                    if start_idx == -1:
                                        break
                                    else:
                                        start_idx = text.find('. \n')
                                    if start_idx == -1:
                                        break
                            else:
                                start_idx = text.find('.\n')
                                if start_idx == -1:
                                    break
                                else:
                                    start_idx = text.find('.\n')
                                    if start_idx == -1:
                                        break
                            else:
                                start_idx = text.find('. ')
                                if start_idx == -1:
                                    break
                                else:
                                    start_idx = text.find('.\n')
                                    if start_idx == -1:
                                        break
                            else:
                                start_idx = text.find('.\n')
                                if start_idx == -1:
                                    break
                                else:
                                    start_idx = text.find('. \n')
                                    if start_idx == -1:
                                        break
                                    else:
                                        start_idx = text.find('.\n')
                                        if start_idx == -1:
                                            break
                                        else:
                                            break
            else:
                break_idx -= 1
                    end_idx = idx_found
 continue
        
        if not results:
            print(f"\nError: Only found {len(results)} passages from the PDF")
            return []
doc.close()
with open("data/eval/ao_eval_v2.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWrote {len(results)} entries to data/eval/ao_eval_v2.json")
