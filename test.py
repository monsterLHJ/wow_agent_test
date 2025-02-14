# 导入必要的库
import openpyxl  # 处理 Excel 文件
import json
import requests  # 用于调用 ChatGPT-4o API
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'test_generator_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestCaseGenerator:
    def __init__(self):
        load_dotenv()
        self.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.api_key = os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError("请在.env文件中设置QWEN_API_KEY")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Algorithm": "qwen-max"
        }

    def read_excel(self, file_path: str) -> List[Dict]:
        """
        读取Excel文件并返回测试用例列表
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"错误：文件 '{file_path}' 不存在！")

            logger.info(f"开始读取文件: {file_path}")
            
            # 使用pandas读取xlsx文件
            import pandas as pd
            df = pd.read_excel(file_path, engine='openpyxl')
            
            # 基本验证
            if df.empty:
                raise ValueError("Excel文件是空的")
            
            if len(df.columns) < 7:
                raise ValueError(f"Excel文件格式不正确：需要7列数据")
            
            # 设置列名
            df.columns = ["一级功能", "二级功能", "优先级", "需求说明", "预置条件", "测试步骤", "预期结果"][:len(df.columns)]
            
            # 处理数据
            test_cases = []
            for index, row in df.iterrows():
                # 跳过包含空值的行
                if row.isna().any():
                    continue

                test_case = {
                    "一级功能": str(row["一级功能"]).strip(),
                    "二级功能": str(row["二级功能"]).strip(),
                    "优先级": str(row["优先级"]).strip(),
                    "需求说明": str(row["需求说明"]).strip(),
                    "预置条件": str(row["预置条件"]).strip(),
                    "测试步骤": str(row["测试步骤"]).strip(),
                    "预期结果": str(row["预期结果"]).strip()
                }
                
                # 只添加所有字段都非空的数据
                if all(test_case.values()):
                    test_cases.append(test_case)

            if not test_cases:
                raise ValueError("没有读取到有效的测试用例数据")
            
            logger.info(f"成功读取到 {len(test_cases)} 条测试用例")
            return test_cases

        except Exception as e:
            logger.error(f"读取Excel文件时出错: {str(e)}")
            raise

    def generate_test_case(self, prompt: str) -> Optional[Dict]:
        """
        调用千问API生成测试用例
        """
        try:
            data = {
                "model": "qwen-max",
                "input": {
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            }
            
            response = requests.post(self.api_url, headers=self.headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            if result.get("status_code") == 200:
                try:
                    return json.loads(result["output"]["text"])
                except json.JSONDecodeError:
                    logger.error(f"API返回的JSON格式无效: {result['output']['text']}")
                    return None
            else:
                logger.error(f"API调用失败: {result.get('message', '未知错误')}")
                return None
                
        except Exception as e:
            logger.error(f"生成测试用例时出错: {str(e)}")
            return None

    def generate_new_test_cases(self, test_cases: List[Dict]) -> List[Dict]:
        """
        批量生成新的测试用例
        """
        new_test_cases = []
        total = len(test_cases)
        
        for i, case in enumerate(test_cases, 1):
            logger.info(f"正在生成第 {i}/{total} 个测试用例...")
            
            prompt = self._create_prompt(case)
            generated_case = self.generate_test_case(prompt)
            
            if generated_case:
                new_test_cases.append(generated_case)
            
        return new_test_cases

    def _create_prompt(self, case: Dict) -> str:
        """
        创建用于生成测试用例的提示
        """
        return f"""
        作为软件测试专家，请基于以下信息生成一个新的测试用例：
        - 一级功能: {case['一级功能']}
        - 二级功能: {case['二级功能']}
        - 优先级: {case['优先级']}
        - 需求说明: {case['需求说明']}
        - 预置条件: {case['预置条件']}
        - 测试步骤: {case['测试步骤']}
        - 预期结果: {case['预期结果']}
        
        请按照以下JSON格式返回：
        {{
            "测试标题": "xxx",
            "测试步骤": "xxx",
            "预期结果": "xxx"
        }}
        
        要求：
        1. 测试步骤要详细且清晰
        2. 预期结果要具体且可验证
        3. 测试标题要简洁明了
        """

    def save_to_excel(self, test_cases: List[Dict], output_file: str) -> None:
        """
        保存测试用例到Excel文件
        """
        try:
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            
            # 添加表头
            headers = ["测试标题", "测试步骤", "预期结果"]
            sheet.append(headers)
            
            # 设置列宽
            for i, header in enumerate(headers, 1):
                sheet.column_dimensions[openpyxl.utils.get_column_letter(i)].width = 40
            
            # 添加数据
            for case in test_cases:
                sheet.append([case["测试标题"], case["测试步骤"], case["预期结果"]])
            
            workbook.save(output_file)
            logger.info(f"测试用例已保存至 {output_file}")
            
        except Exception as e:
            logger.error(f"保存Excel文件时出错: {str(e)}")
            raise

def main():
    try:
        generator = TestCaseGenerator()
        
        input_file = "test_cases.xlsx"
        output_file = f"generated_test_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        logger.info(f"开始读取Excel文件 '{input_file}'...")
        test_cases = generator.read_excel(input_file)
        
        if not test_cases:
            logger.error("没有读取到有效的测试用例")
            return
            
        logger.info(f"成功读取到 {len(test_cases)} 条测试用例")
        
        logger.info("开始生成新的测试用例...")
        new_test_cases = generator.generate_new_test_cases(test_cases)
        
        if not new_test_cases:
            logger.error("没有生成任何新的测试用例")
            return
            
        logger.info(f"成功生成 {len(new_test_cases)} 条新测试用例")
        
        logger.info("保存测试用例到Excel...")
        generator.save_to_excel(new_test_cases, output_file)
        
        logger.info("程序执行完成")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()
