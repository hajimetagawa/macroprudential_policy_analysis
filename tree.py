import os
import matplotlib.pyplot as plt

def get_folder_structure(root_dir, prefix=''):
    structure = ''
    try:
        for item in sorted(os.listdir(root_dir)):
            path = os.path.join(root_dir, item)
            structure += f"{prefix}|-- {item}\n"
            if os.path.isdir(path):
                structure += get_folder_structure(path, prefix + '    ')
    except PermissionError:
        pass  # アクセス権のないフォルダを無視
    return structure

def save_folder_structure_as_image(root_dir, output_file):
    structure = get_folder_structure(root_dir)

    plt.figure(figsize=(12, 12))
    plt.text(0, 1, structure, fontsize=10, va='top', family='monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # カレントディレクトリを対象に構造を取得して出力
    save_folder_structure_as_image('.', 'folder_structure.png')
    print("✅ フォルダ構成を folder_structure.png に保存しました。")
