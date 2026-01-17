# task02.py

from openai import OpenAI
from typing import List, Dict, Tuple, Any
import requests
import json
import json5
import re
import time
import os


class Config:
    LLM_MODEL_ID = os.getenv('LLM_MODEL_ID', 'qwen2.5:3b')
    LLM_API_KEY = os.getenv('LLM-API-KEY', 'localkey')
    LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'http://localhost:11434')
    SERPER_X_API_KEY = os.getenv('SERPER_X_API_KEY', 'wrongn-key')


class BaseModel:
    def __init__(self, api_key: str = '') -> None:
        self.api_key = api_key

    def chat(self, prompt: str, history: List[Dict[str, str]], system_prompt: str = "") -> Tuple[str, List[Dict[str, str]]]:
        """
        基础聊天接口

        Args:
            prompt: 用户输入
            history: 对话历史
            system_prompt: 系统提示

        Returns:
            (模型响应, 更新后的对话历史)
        """
        pass


class Ollama(BaseModel):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key, base_url=f"{config.LLM_BASE_URL}/v1")

    def chat(self, prompt: str, history: List[Dict[str, str]] = [], system_prompt: str = "") -> Tuple[str, List[Dict[str, str]]]:
        """
        与 Ollama API 进行聊天

        Args:
            prompt: 用户输入
            history: 对话历史
            system_prompt: 系统提示

        Returns:
            (模型响应, 更新后的对话历史)
        """
        # 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."}
        ]

        # 添加历史消息
        if history:
            messages.extend(history)

        # 添加当前用户消息
        messages.append({"role": "user", "content": prompt})

        # 调用 API
        response = self.client.chat.completions.create(
            model=config.LLM_MODEL_ID,
            messages=messages,
            temperature=0.6,
            max_tokens=2000,
        )

        model_response = response.choices[0].message.content

        # 更新对话历史
        updated_history = messages.copy()
        updated_history.append({"role": "assistant", "content": model_response})

        return model_response, updated_history


class ReactTools:
    """
    React Agent 工具类

    为 ReAct Agent 提供标准化的工具接口
    """

    def __init__(self) -> None:
        self.toolConfig = self._build_tool_config()

    def _build_tool_config(self) -> List[Dict[str, Any]]:
        """构建工具配置信息"""
        return [
            {
                'name_for_human': '谷歌搜索',
                'name_for_model': 'google_search',
                'description_for_model': '谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。',
                'parameters': [
                    {
                        'name': 'search_query',
                        'description': '搜索关键词或短语',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ],
            }
        ]

    def google_search(self, search_query: str) -> str:
        """执行谷歌搜索

        可在 https://serper.dev/dashboard 申请 api key

        Args:
            search_query: 搜索关键词

        Returns:
            格式化的搜索结果字符串
        """
        url = "https://google.serper.dev/search"

        payload = json.dumps({"q": search_query})
        headers = {
            'X-API-KEY': config.SERPER_X_API_KEY,
            'Content-Type': 'application/json'
        }

        try:
            response = requests.request("POST", url, headers=headers, data=payload).json()
            organic_results = response.get('organic', [])

            # 格式化搜索结果
            formatted_results = []
            for idx, result in enumerate(organic_results[:5], 1):
                title = result.get('title', '无标题')
                snippet = result.get('snippet', '无描述')
                link = result.get('link', '')
                formatted_results.append(f"{idx}. **{title}**\n   {snippet}\n   链接: {link}")

            return "\n\n".join(formatted_results) if formatted_results else "未找到相关结果"

        except Exception as e:
            return f"搜索时出现错误: {str(e)}"

    def get_available_tools(self) -> List[str]:
        return [tool['name_for_model'] for tool in self.toolConfig]

    def get_tool_description(self, tool_name: str) -> str:
        for tool in self.toolConfig:
            if tool['name_for_model'] == tool_name:
                return tool['description_for_model']
        return "未知工具"


class ReactAgent:
    def __init__(self, api_key: str = '') -> None:
        """初始化 React Agent"""
        self.api_key = api_key or config.LLM_API_KEY
        self.tools = ReactTools()
        self.model = Ollama(api_key)
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """构建 ReAct 系统提示"""
        # 组合工具描述和 ReAct 模式指导
        prompt = f"""现在时间是 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}。
你是一位智能助手，可以使用以下工具来回答问题：

{tool_descriptions}

请遵循以下 ReAct 模式：

思考：分析问题和需要使用的工具
行动：选择工具 [google_search] 中的一个
行动输入：提供工具的参数
观察：工具返回的结果

你可以重复以上循环，直到获得足够的信息来回答问题。

最终答案：基于所有信息给出最终答案

开始！"""
        return prompt

    def _parse_action(self, text: str) -> tuple[str, dict]:
        """解析模型输出中的行动和参数"""
        # 使用正则表达式提取行动和参数
        action_pattern = r"行动[:：]\s*(\w+)"
        action_input_pattern = r"行动输入[:：]\s*({.*?}|[^\n]*)"

        action_match = re.search(action_pattern, text, re.IGNORECASE)
        action_input_match = re.search(action_input_pattern, text, re.DOTALL)

        action = action_match.group(1).strip() if action_match else ""
        action_input_str = action_input_match.group(1).strip() if action_input_match else ""

        # 清理和解析JSON
        action_input_dict = {}
        if action_input_str:
            try:
                # 尝试解析为JSON对象
                action_input_str = action_input_str.strip()
                if action_input_str.startswith('{') and action_input_str.endswith('}'):
                    action_input_dict = json5.loads(action_input_str)
                else:
                    # 如果不是JSON格式，尝试解析为简单字符串参数
                    action_input_dict = {"search_query": action_input_str.strip('"\'')}
            except Exception as e:
                if verbose:
                    print(f"[ReAct Agent] 解析参数失败，使用字符串作为搜索查询: {e}")
                action_input_dict = {"search_query": action_input_str.strip('"\'')}

        return action, action_input_dict

    def _execute_action(self, action: str, action_input: dict) -> str:
        """执行指定的工具行动"""
        # 调用对应工具并返回结果
        if action == "google_search":
            search_query = action_input.get("search_query", "")
            if search_query:
                results = self.tools.google_search(search_query)
                return f"观察：搜索完成，结果如下：\n{results}"
            else:
                return "观察：缺少搜索查询参数"

        return f"观察：未知行动 '{action}'"

    def _format_response(self, response_text: str) -> str:
        """格式化最终响应"""
        if "最终答案：" in response_text:
            return response_text.split("最终答案：")[-1].strip()
        return response_text

    def run(self, query: str, max_iterations: int = 3, verbose: bool = True) -> str:
        """运行 ReAct Agent 主循环"""
        # 实现思考-行动-观察循环
        conversation_history = []
        current_text = f"问题：{query}"

        # 绿色ANSI颜色代码
        GREEN = '\033[92m'
        RESET = '\033[0m'

        if verbose:
            print(f"{GREEN}[ReAct Agent] 开始处理问题: {query}{RESET}")

        for iteration in range(max_iterations):
            if verbose:
                print(f"{GREEN}[ReAct Agent] 第 {iteration + 1} 次思考...{RESET}")

            # 获取模型响应
            response, history = self.model.chat(
                current_text,
                conversation_history,
                self.system_prompt
            )

            if verbose:
                print(f"{GREEN}[ReAct Agent] 模型响应:\n{response}{RESET}")

            # 解析行动
            action, action_input = self._parse_action(response, verbose=verbose)

            if not action or action == "最终答案":
                final_answer = self._format_response(response)
                if verbose:
                    print(f"{GREEN}[ReAct Agent] 无需进一步行动，返回最终答案{RESET}")
                return final_answer

            if verbose:
                print(f"{GREEN}[ReAct Agent] 执行行动: {action} | 参数: {action_input}{RESET}")

            # 执行行动
            observation = self._execute_action(action, action_input)

            if verbose:
                print(f"{GREEN}[ReAct Agent] 观察结果:\n{observation}{RESET}")

            # 更新当前文本以继续对话
            current_text = f"{response}\n观察结果:{observation}\n"
            conversation_history = history

        # 达到最大迭代次数，返回当前响应
        if verbose:
            print(f"{GREEN}[ReAct Agent] 达到最大迭代次数，返回当前响应{RESET}")
        return self._format_response(response)

#

if __name__ == '__main__':
    config = Config()
    print(config)
    agent = ReactAgent()
    response = agent.run("最近两周内港股上市的科技公司有哪些？", max_iterations=3, verbose=True)
    print("最终答案：", response)
