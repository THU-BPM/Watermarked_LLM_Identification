# Copyright 2024 THU BPM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

def add_license_header(directory):
    license_header = '''# Copyright 2024 THU BPM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''

    # 遍历目录下的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                # 读取原文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查文件是否已经包含版权声明
                if not content.startswith('# Copyright 2024 THU BPM'):
                    # 添加版权声明并写入文件
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(license_header + '\n' + content)
                    print(f'Added license header to: {file_path}')
                else:
                    print(f'License header already exists in: {file_path}')

# 使用示例
directory_path = '.'  # 当前目录，你可以修改为其他目录路径
add_license_header(directory_path)