import os

# 要收集的目录
TARGET_DIRS = ["configs", "src"]
# 输出的 markdown 文件
OUTPUT_FILE = "project_code_dump.md"


def dump_files_to_md(target_dirs, output_file):
    with open(output_file, "w", encoding="utf-8") as out_f:
        for d in target_dirs:
            if not os.path.exists(d):
                print(f"目录 {d} 不存在，跳过。")
                continue
            for root, _, files in os.walk(d):
                for file in files:
                    file_path = os.path.join(root, file)
                    # 只处理文本文件
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                    except Exception as e:
                        print(f"跳过 {file_path} (无法读取: {e})")
                        continue

                    # 写入 markdown
                    rel_path = os.path.relpath(file_path)
                    out_f.write(f"## {rel_path}\n\n")
                    out_f.write(
                        "```python\n" if file.endswith(".py") else "```yaml\n" if file.endswith(".yaml") else "```\n")
                    out_f.write(content)
                    out_f.write("\n```\n\n")

    print(f"✅ 已生成 {output_file}")


if __name__ == "__main__":
    dump_files_to_md(TARGET_DIRS, OUTPUT_FILE)
