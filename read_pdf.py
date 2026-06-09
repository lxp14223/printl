import fitz

pdf_path = r'e:\pythonProject\machine\print\print_l\基于视觉检测的凹版印刷装饰纸缺陷识别与质量管控_王峰.pdf'
doc = fitz.open(pdf_path)
print(f'PDF页数: {len(doc)}')

all_text = []
for i, page in enumerate(doc):
    text = page.get_text()
    if text.strip():
        all_text.append(f'\n--- 第{i+1}页 ---\n{text}')

doc.close()

# 写入文件
with open(r'e:\pythonProject\machine\print\print_l\pdf_content.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(all_text))

print('PDF内容已保存到 pdf_content.txt')
