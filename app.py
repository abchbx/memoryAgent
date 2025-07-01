# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# 您的专属记忆助理 V4.6 - 上传属性修复版
#
# 更新日志 (V4.6):
# - BUG修复: 修复了因使用错误的 `UploadedFile.id` 属性导致的 `AttributeError`。现在使用正确的 `UploadedFile.file_id`。
#
# 更新日志 (V4.5):
# - BUG修复: 修复了文件上传后因 `st.rerun()` 导致的重复处理问题。
#
# 更新日志 (V4.4):
# - 核心逻辑重构: 区分了“永久性静态事实”和“临时性事实”。
# - 登录优化: 登录时只会清理过期的“临时性事实”，用户的核心偏好和身份信息将被永久保留。
# - AI能力升级: 优化了记忆提取的Prompt，使AI能更准确地分类不同类型的事实。
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 步骤 1: 导入所有必要的库
# -----------------------------------------------------------------------------
import os
import json
import logging
import time
import datetime
import pytz # 用于处理时区
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# -----------------------------------------------------------------------------
# 步骤 2: 后端逻辑代码 (V4.6 - 上传属性修复版)
# -----------------------------------------------------------------------------

# --- 日志记录配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 核心配置类 ---
class Config:
    """集中管理所有配置"""
    load_dotenv()
    ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
    LLM_MODEL = "glm-4-flash"
    EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
    DB_PATH = "/workspace/memoryAgent/user_centric_db_v4.6" # V4.6: 数据库路径更新
    CHAT_COLLECTION_NAME = "user_chat_history"
    FACT_MEMORY_COLLECTION_NAME = "user_fact_memory"
    EVENT_MEMORY_COLLECTION_NAME = "user_event_memory"
    RAG_COLLECTION_NAME = "user_rag_documents"
    TIMEZONE = "Asia/Shanghai"

    SYSTEM_PROMPT_TEMPLATE = """
    你是为用户 {user_id} 服务的顶级个人智能助手。

    # 当前时间: {current_time}

    # 关于用户 {user_id} 的已知事实 (你的静态记忆):
    {long_term_memory}

    # 关于用户 {user_id} 的相关事件与计划 (你的动态记忆):
    ## 未来计划 (按时间正序):
    {future_events}
    ## 最近发生的事件 (按时间倒序):
    {past_events}

    # 你的工作流程:
    1.  **深入理解**: 结合当前时间，分析用户的最新问题。
    2.  **整合信息**: 我会为你提供用户的长期事实记忆、按时间排序的事件记忆、相关的历史对话、以及知识库资料。你必须将这些信息全部整合，形成对上下文的完整理解。
    3.  **优先使用知识库**: 如果知识库信息与问题直接相关，优先基于这些信息回答。
    4.  **个性化回答**: 基于所有信息，为用户 {user_id} 生成一个富有洞察力、连贯且个性化的回答。
    """
    
    MEMORY_EXTRACTION_INTERVAL = 4

# --- 文本分割器 ---
def simple_text_splitter(text: str, max_chunk_size: int = 500) -> list[str]:
    """一个简单的文本分割器，按句子分割"""
    sentences = text.replace("\n", " ").replace("\r", " ").split('。')
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if not sentence.strip(): continue
        sentence += "。"
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks

# --- 数据库管理类 ---
class ChatHistoryDB:
    def __init__(self, config: Config):
        self.config = config
        self.tz = pytz.timezone(config.TIMEZONE)
        try:
            os.makedirs(config.DB_PATH, exist_ok=True)
            self.db_client = chromadb.PersistentClient(path=self.config.DB_PATH)
            self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config.EMBEDDING_MODEL
            )
            self.chat_collection = self.db_client.get_or_create_collection(name=self.config.CHAT_COLLECTION_NAME, embedding_function=self.embedding_func)
            self.fact_memory_collection = self.db_client.get_or_create_collection(name=self.config.FACT_MEMORY_COLLECTION_NAME, embedding_function=self.embedding_func)
            self.event_memory_collection = self.db_client.get_or_create_collection(name=self.config.EVENT_MEMORY_COLLECTION_NAME, embedding_function=self.embedding_func)
            self.rag_collection = self.db_client.get_or_create_collection(name=self.config.RAG_COLLECTION_NAME, embedding_function=self.embedding_func)
            logging.info(f"数据库初始化成功: {config.DB_PATH}")
        except Exception as e:
            logging.error(f"初始化 ChromaDB 失败: {e}")
            st.error(f"数据库初始化失败: {e}")
            raise

    def add_document_to_rag(self, user_id: str, file_name: str, file_content: str, progress_callback=None):
        """将文档添加到RAG知识库，并提供清晰的进度回调"""
        if progress_callback: progress_callback(0, "步骤 1/2: 正在分割文件...")
        chunks = simple_text_splitter(file_content)
        if not chunks:
            if progress_callback: progress_callback(100, "文件内容为空，已跳过。")
            return

        total_chunks = len(chunks)
        logging.info(f"文件 '{file_name}' 被分割成 {total_chunks} 个片段。")
        
        if progress_callback: progress_callback(5, f"步骤 2/2: 准备计算向量... (共 {total_chunks} 块)")
        
        batch_size = 32
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_ids = [f"rag_{user_id}_{file_name}_{time.time()}_{i+j}" for j in range(len(batch_chunks))]
            batch_metadatas = [{"user_id": user_id, "source": file_name} for _ in batch_chunks]
            
            self.rag_collection.add(ids=batch_ids, documents=batch_chunks, metadatas=batch_metadatas)
            
            if progress_callback:
                processed_count = i + len(batch_chunks)
                percentage = 5 + int((processed_count / total_chunks) * 90)
                status_text = f"步骤 2/2: 正在计算向量... ({processed_count}/{total_chunks})"
                progress_callback(min(percentage, 95), status_text)
        
        if progress_callback: progress_callback(100, "知识库学习完成！")
        logging.info(f"文件 '{file_name}' 已成功添加至知识库。")

    def save_message(self, user_id: str, message: dict):
        if not all(k in message for k in ['role', 'content']) or not message.get('content'): return
        timestamp = time.time()
        doc_id = f"msg_{user_id}_{timestamp}"
        self.chat_collection.add(ids=[doc_id], documents=[message['content']], metadatas=[{"user_id": user_id, "role": message['role'], "timestamp": timestamp}])

    def load_history_by_user(self, user_id: str) -> list:
        if not user_id: return []
        results = self.chat_collection.get(where={"user_id": user_id})
        if not results['ids']: return []
        packaged_messages = sorted([{"doc": results['documents'][i], "meta": results['metadatas'][i]} for i in range(len(results['ids']))], key=lambda m: m['meta']['timestamp'])
        return [{"role": msg['meta']['role'], "content": msg['doc']} for msg in packaged_messages]

    def clear_user_chat_history(self, user_id: str):
        if self.chat_collection.get(where={"user_id": user_id})['ids']:
            self.chat_collection.delete(where={"user_id": user_id})
            logging.info(f"已清空用户 {user_id} 的对话历史。")

    def clear_old_temporal_fact_memory(self, user_id: str):
        """清除指定用户今天之前的【临时】事实记忆，保留永久性静态事实。"""
        logging.info(f"开始为用户 {user_id} 清理旧的【临时】事实记忆...")
        
        today_start = datetime.datetime.now(self.tz).replace(hour=0, minute=0, second=0, microsecond=0)
        today_start_timestamp = today_start.timestamp()

        results = self.fact_memory_collection.get(where={"user_id": user_id})
        if not results['ids']:
            logging.info(f"用户 {user_id} 没有事实记忆可清理。")
            return

        ids_to_delete = [
            results['ids'][i] 
            for i, meta in enumerate(results['metadatas']) 
            # 核心条件: 只删除 (is_permanent不为True) 且 (时间戳早于今天) 的记忆
            if meta.get('is_permanent') is not True and meta.get('timestamp') and meta['timestamp'] < today_start_timestamp
        ]

        if ids_to_delete:
            logging.info(f"为用户 {user_id} 找到 {len(ids_to_delete)} 条旧的临时事实记忆，准备删除...")
            self.fact_memory_collection.delete(ids=ids_to_delete)
            logging.info(f"已成功为用户 {user_id} 清理了 {len(ids_to_delete)} 条旧的临时事实记忆。")
        else:
            logging.info(f"用户 {user_id} 没有今天之前的旧临时事实记忆可清理。")
            
    def save_structured_memory(self, user_id: str, memory_data: dict):
        """保存结构化的记忆，区分永久事实、临时事实和动态事件"""

        def _save_facts(facts: dict, is_permanent: bool):
            if not (facts and isinstance(facts, dict)):
                return
            
            fact_type_str = "永久" if is_permanent else "临时"
            logging.info(f"正在为用户 {user_id} 保存或更新 {len(facts)} 条{fact_type_str}事实记忆...")
            
            items_to_save = []
            for key, value in facts.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        items_to_save.append((f"{key}_{sub_key}", sub_value))
                else:
                    items_to_save.append((key, value))
            
            for key, value in items_to_save:
                doc_id = f"fact_{user_id}_{key}"
                self.fact_memory_collection.upsert(
                    ids=[doc_id],
                    documents=[f"用户的个人信息：{key} 是 {value}。"],
                    metadatas=[{
                        "user_id": user_id, 
                        "key": key, 
                        "timestamp": time.time(),
                        "is_permanent": is_permanent  # 新增的关键元数据字段
                    }]
                )

        # 处理永久性事实
        permanent_facts = memory_data.get('permanent_facts', {})
        _save_facts(permanent_facts, is_permanent=True)

        # 处理临时性事实 (并兼容旧的 static_facts 键)
        temporal_facts = memory_data.get('temporal_facts', {})
        if not temporal_facts and 'static_facts' in memory_data:
             temporal_facts = memory_data.get('static_facts', {}) # 向后兼容
        _save_facts(temporal_facts, is_permanent=False)

        # 事件处理逻辑保持不变
        events = memory_data.get('events', [])
        if events and isinstance(events, list):
            logging.info(f"正在为用户 {user_id} 保存 {len(events)} 条事件记忆...")
            for event in events:
                if isinstance(event, dict) and 'description' in event and 'event_time_iso' in event:
                    description = event['description']
                    event_time_iso = event['event_time_iso']
                    event_time_desc = event.get('event_time_desc', '未知时间')
                    
                    doc_id = f"event_{user_id}_{time.time()}"
                    self.event_memory_collection.add(
                        ids=[doc_id],
                        documents=[f"事件: {description} (时间: {event_time_desc})"],
                        metadatas={
                            "user_id": user_id, 
                            "event_time_iso": event_time_iso,
                            "event_time_desc": event_time_desc,
                            "saved_at": time.time()
                        }
                    )

    def load_fact_memory(self, user_id: str, top_k: int = 20) -> str:
        """加载静态事实记忆"""
        results = self.fact_memory_collection.get(where={"user_id": user_id}, limit=top_k)
        return "\n".join(f"- {doc}" for doc in results.get('documents', [])) or "暂无"
    
    def load_event_memory(self, user_id: str, query: str = None) -> tuple[str, str]:
        """
        加载与用户相关的事件记忆，精确区分未来和过去。
        返回一个元组: (未来事件字符串, 过去事件字符串)
        """
        if not user_id: return "暂无", "暂无"
        
        all_events_result = self.event_memory_collection.get(where={"user_id": user_id})
        if not all_events_result['ids']: return "暂无", "暂无"

        now = datetime.datetime.now(self.tz)
        future_events, past_events = [], []

        for i, meta in enumerate(all_events_result['metadatas']):
            event_time_iso = meta.get('event_time_iso')
            if not event_time_iso: continue
            
            try:
                if 'Z' in event_time_iso or '+' in event_time_iso or '-' in event_time_iso[10:]:
                    event_dt = datetime.datetime.fromisoformat(event_time_iso)
                else:
                    event_dt = self.tz.localize(datetime.datetime.fromisoformat(event_time_iso))
            except (ValueError, TypeError):
                continue

            event_record = (event_dt, all_events_result['documents'][i])
            if event_dt > now:
                future_events.append(event_record)
            else:
                past_events.append(event_record)

        future_events.sort(key=lambda x: x[0])
        past_events.sort(key=lambda x: x[0], reverse=True)

        future_str = "\n".join(f"- {doc} (时间: {dt.strftime('%Y-%m-%d %H:%M')})" for dt, doc in future_events) or "暂无"
        past_str = "\n".join(f"- {doc} (时间: {dt.strftime('%Y-%m-%d %H:%M')})" for dt, doc in past_events[:5]) or "暂无"

        return future_str, past_str

    def get_all_event_memory_for_display(self, user_id: str) -> str:
        """获取所有事件记忆用于UI展示，并按时间排序"""
        results = self.event_memory_collection.get(where={"user_id": user_id})
        if not results['ids']: return "暂无事件记忆。"
        
        events_with_time = []
        for i, meta in enumerate(results['metadatas']):
            event_time_iso = meta.get('event_time_iso')
            doc = results['documents'][i]
            try:
                dt = datetime.datetime.fromisoformat(event_time_iso.replace('Z', '+00:00')) if event_time_iso else datetime.datetime.min
                events_with_time.append((dt, doc))
            except ValueError:
                events_with_time.append((datetime.datetime.min, f"{doc} (时间解析失败: {event_time_iso})"))
        
        events_with_time.sort(key=lambda x: x[0], reverse=True)
        return "\n".join(f"{doc}" for dt, doc in events_with_time)

    def clear_user_rag_documents(self, user_id: str):
        if self.rag_collection.get(where={"user_id": user_id})['ids']:
            self.rag_collection.delete(where={"user_id": user_id})
            logging.info(f"已清空用户 {user_id} 的知识库。")

    def get_rag_file_list(self, user_id: str) -> list[str]:
        results = self.rag_collection.get(where={"user_id": user_id})
        if not results['ids']: return []
        return sorted(list({meta['source'] for meta in results['metadatas'] if 'source' in meta}))

    @st.cache_data(show_spinner=False)
    def query_rag_documents(_self, user_id: str, query: str, top_k: int = 3) -> str:
        if not _self.rag_collection.get(where={"user_id": user_id}, limit=1)['ids']: return ""
        results = _self.rag_collection.query(query_texts=[query], where={"user_id": user_id}, n_results=top_k)
        retrieved_docs = results.get("documents", [[]])[0]
        if not retrieved_docs: return ""
        return f"从你的知识库中找到以下相关信息：\n" + "\n".join(f"- {doc}" for doc in retrieved_docs)
        
    @st.cache_data(show_spinner=False)
    def query_recent_discussions(_self, user_id: str, query: str, top_k: int = 3) -> str:
        if not _self.chat_collection.get(where={"user_id": user_id}, limit=1)['ids']: return "该用户没有任何历史对话记录。"
        results = _self.chat_collection.query(query_texts=[query], where={"user_id": user_id}, n_results=top_k)
        retrieved_docs = results.get("documents", [[]])[0]
        if not retrieved_docs: return "在你的历史记录中，没有找到与当前问题相关的内容。"
        return f"你回忆起了以下可能相关的历史对话内容：\n" + "\n".join(f"- \"{doc}\"" for doc in retrieved_docs)

# --- 智能代理类 ---
class ChatAgent:
    def __init__(self, config: Config, db_manager: ChatHistoryDB, api_key: str, user_id: str):
        self.config, self.db_manager, self.api_key, self.user_id = config, db_manager, api_key, user_id
        self.tz = pytz.timezone(config.TIMEZONE)
        if not api_key: raise ValueError("必须提供 ZHIPU_API_KEY。")
        self.client = OpenAI(api_key=api_key, base_url=self.config.ZHIPU_BASE_URL)
        self.refresh_agent_state()

    def refresh_agent_state(self, query: str = None):
        """刷新代理状态，加载所有类型的记忆并注入当前时间"""
        fact_memory = self.db_manager.load_fact_memory(self.user_id)
        future_events, past_events = self.db_manager.load_event_memory(self.user_id, query=query)
        
        now_time = datetime.datetime.now(self.tz)
        current_time_str = now_time.strftime('%Y-%m-%d %H:%M:%S %Z')

        system_prompt = self.config.SYSTEM_PROMPT_TEMPLATE.format(
            user_id=self.user_id,
            current_time=current_time_str,
            long_term_memory=fact_memory,
            future_events=future_events,
            past_events=past_events
        )
        
        self.messages = self.db_manager.load_history_by_user(self.user_id)
        if not self.messages or self.messages[0]['role'] != 'system':
            self.messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            self.messages[0]['content'] = system_prompt

    def run(self, user_input: str):
        self.refresh_agent_state(query=user_input)
        
        context_from_rag = self.db_manager.query_rag_documents(self.user_id, user_input)
        context_from_history = self.db_manager.query_recent_discussions(self.user_id, user_input)
        
        messages_for_llm = list(self.messages)
        if context_from_rag: messages_for_llm.append({"role": "system", "content": f"补充信息-知识库检索:\n{context_from_rag}"})
        if context_from_history: messages_for_llm.append({"role": "system", "content": f"补充信息-历史对话回顾:\n{context_from_history}"})
        
        messages_for_llm.append({"role": "user", "content": user_input})
        
        try:
            response = self.client.chat.completions.create(model=self.config.LLM_MODEL, messages=messages_for_llm)
            final_response = response.choices[0].message.content or "抱歉，我不知道如何回复。"
        except Exception as e:
            logging.error(f"调用LLM API失败: {e}")
            final_response = f"抱歉，出错了: {e}"
            
        user_message = {"role": "user", "content": user_input}
        assistant_message = {"role": "assistant", "content": final_response}
        
        self.db_manager.save_message(self.user_id, user_message)
        self.db_manager.save_message(self.user_id, assistant_message)
        
        self.messages.extend([user_message, assistant_message])
        
        user_message_count = sum(1 for msg in self.messages if msg['role'] == 'user')
        if user_message_count > 0 and user_message_count % self.config.MEMORY_EXTRACTION_INTERVAL == 0:
            self.extract_and_save_memory()
            
        return final_response

    def extract_and_save_memory(self):
        conversation = [msg for msg in self.messages if msg['role'] in ['user', 'assistant']]
        if len(conversation) < 2: return False
        
        full_chat_content = "\n".join([f"{m['role']}: {m['content']}" for m in conversation])
        
        now_time = datetime.datetime.now(self.tz)
        current_time_iso = now_time.isoformat()

        # 更新后的Prompt，要求LLM区分永久性和临时性事实
        memory_prompt = f"""
        请仔细阅读用户 {self.user_id} 的对话，并以JSON格式，提炼出三种信息：
        1.  `permanent_facts`: 关于用户的【核心事实】和【长期偏好】。这些信息非常稳定，几乎不会改变（例如：姓名、职业、出生地、基本价值观、不喜欢的食物）。
        2.  `temporal_facts`: 关于用户的【临时状态】或【近期事实】。这些信息在短期内有效，但可能很快过时（例如：今天的心情、最近完成的任务、本周的目标）。
        3.  `events`: 对话中提到的【未来计划】或【已经发生的具体事件】。

        **重要规则**:
        - 对于每个`event`，必须包含三个字段：
          1. `description`: 事件的文字描述。
          2. `event_time_desc`: 对话中提到的原始时间描述（如“明天下午”）。
          3. `event_time_iso`: **必须基于当前时间 `{current_time_iso}` 将 `event_time_desc` 解析为标准的 ISO 8601 格式时间戳 (YYYY-MM-DDTHH:MM:SS±HH:MM)**。
        - `permanent_facts` 和 `temporal_facts` 都应该是键值对形式的JSON对象。
        - 如果对话中没有发现任何特定类型的信息，请让其对应的值为空的JSON对象或数组。例如: {{"permanent_facts": {{}}, "temporal_facts": {{"mood": "happy"}}, "events": []}}

        对话内容:
        ---
        {full_chat_content}
        ---
        提取的JSON:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[{"role": "user", "content": memory_prompt}],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if content and (extracted_data := json.loads(content)):
                self.db_manager.save_structured_memory(self.user_id, extracted_data)
                self.refresh_agent_state()
                return True
            return False
        except Exception as e:
            logging.error(f"解析或保存长期记忆失败: {e}")
            return False

# -----------------------------------------------------------------------------
# 步骤 3: Streamlit 前端界面 (V4.6 - 上传属性修复版)
# -----------------------------------------------------------------------------

st.set_page_config(page_title="您的专属记忆助理 V4.6", page_icon="🧠", layout="centered")

@st.cache_resource
def get_core_services():
    config = Config()
    db_manager = ChatHistoryDB(config)
    return config, db_manager

config, db_manager = get_core_services()
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    st.error("未找到 ZHIPUAI_API_KEY 环境变量，请配置。")
    st.stop()
    
if "logged_in_user_id" not in st.session_state: st.session_state.logged_in_user_id = None
if "agent" not in st.session_state: st.session_state.agent = None
if "last_uploaded_file_id" not in st.session_state: st.session_state.last_uploaded_file_id = None

# --- 侧边栏 ---
with st.sidebar:
    st.header("👤 用户中心")
    user_id_input = st.text_input("请输入您的用户ID", key="user_id_input", placeholder="例如: zhangsan")
    if st.button("登录 / 切换用户", key="login_button"):
        if user_id_input:
            if st.session_state.agent and st.session_state.logged_in_user_id != user_id_input:
                with st.spinner("沉淀最终记忆..."): st.session_state.agent.extract_and_save_memory()
            
            st.session_state.logged_in_user_id = user_id_input
            
            # 更新了提示文本和调用的函数
            with st.spinner(f"正在为您清理过期的临时记忆..."):
                db_manager.clear_old_temporal_fact_memory(user_id_input)
                time.sleep(1)

            st.session_state.agent = None
            st.toast(f"欢迎回来, {user_id_input}！您的记忆已刷新。", icon="✅")
            st.rerun()
        else: st.warning("请输入一个用户ID。")

    if st.session_state.logged_in_user_id:
        current_user_id = st.session_state.logged_in_user_id
        st.markdown("---")
        
        st.header("📚 知识库 (RAG)")
        uploaded_file = st.file_uploader("上传知识文件 (.txt/.md)", type=['txt', 'md'], key=f"uploader_{current_user_id}")
        
        # 检查是否是新上传的文件，且尚未被处理
        if uploaded_file is not None and uploaded_file.file_id != st.session_state.get('last_uploaded_file_id'):
            progress_container = st.empty()
            try:
                def update_progress(percent, message):
                    progress_container.progress(percent, text=message)
                
                content = uploaded_file.getvalue().decode("utf-8")
                db_manager.add_document_to_rag(current_user_id, uploaded_file.name, content, progress_callback=update_progress)
                
                # 标记文件已处理
                st.session_state.last_uploaded_file_id = uploaded_file.file_id
                
                time.sleep(1) # 短暂显示完成状态
                progress_container.empty()
                st.toast(f"文件 '{uploaded_file.name}' 已学习完成！", icon="✅")
                st.rerun() # 安全地刷新界面
            except Exception as e:
                # 即使失败也要标记，防止无限重试
                st.session_state.last_uploaded_file_id = uploaded_file.file_id
                progress_container.empty()
                st.error(f"处理文件失败: {e}")
        
        rag_files = db_manager.get_rag_file_list(current_user_id)
        if rag_files:
            with st.expander("查看已上传的文件", expanded=False):
                for f in rag_files: st.caption(f)
            if st.button("🗑️ 清空所有知识库文件", key="clear_rag"):
                with st.spinner("清空知识库..."):
                    db_manager.clear_user_rag_documents(current_user_id)
                    st.cache_data.clear()
                st.toast("知识库已清空！", icon="🗑️")
                st.rerun()

        st.markdown("---")
        st.header("🛠️ 记忆工具箱")
        
        if st.button("🔄 清理应用缓存", key="clear_app_cache", help="当应用行为异常或代码更新后未生效时，可尝试清理缓存。"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.session_state.agent = None
            st.toast("应用缓存已清理！应用将重新加载。", icon="♻️")
            time.sleep(1)
            st.rerun()

        if st.button("🧠 主动提炼记忆", key="extract_memory"):
            if st.session_state.agent:
                with st.spinner("⏳ 正在分析和沉淀记忆..."):
                    saved = st.session_state.agent.extract_and_save_memory()
                if saved:
                    st.success("✅ 记忆已更新！")
                else:
                    st.info("🤷‍ 未发现新的可记忆信息。")
                time.sleep(2)
                st.rerun()

        if st.button("🧹 清理当前对话", key="clear_chat"):
            if st.session_state.agent:
                with st.spinner("保存最后的回忆..."): st.session_state.agent.extract_and_save_memory()
                with st.spinner("清空对话历史..."):
                    db_manager.clear_user_chat_history(current_user_id)
                    st.cache_data.clear()
                st.session_state.agent = None
                st.toast("对话历史已清空！", icon="🗑️")
                st.rerun()
        
        with st.expander("👀 查看我的事实记忆 (包含永久和临时)"):
            memory_content = db_manager.load_fact_memory(current_user_id)
            st.code(memory_content, language=None) if memory_content.strip() and memory_content != "暂无" else st.info("暂无事实记忆。")

        with st.expander("📅 查看我的事件记忆 (按时间倒序)"):
            event_memory_content = db_manager.get_all_event_memory_for_display(current_user_id)
            st.code(event_memory_content, language=None) if event_memory_content.strip() and "暂无" not in event_memory_content else st.info("暂无事件记忆。")

    else: st.caption("请先登录以使用全部功能。")

# --- 主聊天界面 ---
st.title("🧠 您的专属记忆助理 V4.6")
st.caption("现在我能区分并永久保留您的核心记忆了！每次登录仅会清理过期的临时记忆。")
if not st.session_state.logged_in_user_id:
    st.info("👈 请在左侧边栏输入用户ID并登录。")
    st.stop()
try:
    if st.session_state.agent is None:
        st.session_state.agent = ChatAgent(config, db_manager, api_key, st.session_state.logged_in_user_id)
except Exception as e:
    st.error(f"初始化助理出错: {e}")
    st.stop()

for message in st.session_state.agent.messages:
    if message["role"] in ["user", "assistant"]:
        with st.chat_message(message["role"]): st.markdown(message["content"])

if prompt := st.chat_input(f"您好, {st.session_state.logged_in_user_id}, 有何贵干?"):
    st.chat_message("user").markdown(prompt)
    with st.spinner("思考中..."):
        response = st.session_state.agent.run(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.rerun()
