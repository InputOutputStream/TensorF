import os
import sys
import ollama

# --- CONFIGURATION ---
MODEL_NAME = 'novaforgeai/qwen2.5:3b-optimized' 
OUTPUT_FILE = 'API_REFERENCE.md'
CODE_DIR = '.'  # Scans current directory
# ---------------------

def process_codebase(directory):
    # Initialize/Clear the markdown file with a header
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        out_f.write("# TensorF API Documentation\n\n")
        out_f.write(f"*Generated locally via Ollama using `{MODEL_NAME}`*\n\n---\n\n")

    files_processed = 0

    # Walk through the repo
    for root, _, files in os.walk(directory):
        # FIX: Ignore actual hidden directories (like .git), but do NOT skip '.' or paths like './include'
        if any(part.startswith('.') for part in root.split(os.sep) if part not in ('.', '..')) or 'build' in root:
            continue
            
        for file in files:
            # Case-insensitive check for all variations of C++ files
            if file.lower().endswith(('.cpp', '.h', '.hpp', '.cc', '.cxx')):
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, directory)
                
                print(f"🔎 Extracting comments from: {rel_path}...")
                files_processed += 1
                
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if not content.strip():
                    print(f"   ⚠️ Skipping {rel_path} (File is empty)")
                    continue

                # Prompt tailored to process a single file cleanly
                prompt = f"""
                    You are an expert C++ documentation extractor. Analyze the following file: `{rel_path}`.

                    Identify all classes, structs, and functions. For each entity, extract only the plain text comments (`//` or `/* */`) that live directly above its definition. Format it into clean Markdown.

                    Rules:
                    1. Do NOT document inner loop tracking or inline logic comments inside function bodies. 
                    2. Only focus on comments documenting class, struct, or function signatures.
                    3. If a function has no comments above it, just list its name and signature with "No documentation provided."

                    File Content:
                    ```cpp
                    {content}

                    ```
                    """
                    
                try:
                    response = ollama.generate(
                    model=MODEL_NAME,
                    prompt=prompt,
                    options={
                    "num_ctx": 16384,  # Safe headroom for single files
                    "temperature": 0.1 # Keep it strict and deterministic
                    }
                    )

                    # Stream the output directly into your master documentation file
                    with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
                        out_f.write(f"## File: `{rel_path}`\n\n")
                        out_f.write(response['response'])
                        out_f.write("\n\n---\n\n")
                                                
                except Exception as e:
                    print(f"❌ Error processing {rel_path}: {e}")

                print(f"\n🎉 Success! Processed {files_processed} files. Combined documentation saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    print(f"🚀 Starting documentation generation with {MODEL_NAME}...")
    process_codebase(CODE_DIR)

