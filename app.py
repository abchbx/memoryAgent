# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# 步骤 1: 导入所有必要的库
# -----------------------------------------------------------------------------
import os
import json
import logging
import time
import datetime
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# -----------------------------------------------------------------------------
# 步骤 2: 后端逻辑代码 (V3.1 - 缓存修复版)
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
    DB_PATH = "/workspace/memoryAgent/user_centric_db_v3"
    CHAT_COLLECTION_NAME = "user_chat_history"
    FACT_MEMORY_COLLECTION_NAME = "user_fact_memory" # V3.0: 名字变更，更清晰
    EVENT_MEMORY_COLLECTION_NAME = "user_event_memory" # V3.0: 新增事件记忆集合
    RAG_COLLECTION_NAME = "user_rag_documents"

    # V3.0: System-Prompt 升级，增加了事件记忆模块
    SYSTEM_PROMPT_TEMPLATE = """
    你是为用户 {user_id} 服务的顶级个人智能助手，拥有卓越的记忆、推理和知识库查询能力。

    # 关于用户 {user_id} 的已知事实 (你的静态记忆):
    {long_term_memory}

    # 关于用户 {user_id} 的相关事件与计划 (你的动态记忆):
    {event_memory}

    # 你的工作流程:
    1.  **深入理解**: 分析用户的最新问题。
    2.  **结合记忆与知识**: 我会为你提供三类信息：用户的长期记忆(事实和事件)、相关的历史对话、以及从用户上传的知识库中检索到的相关资料。你必须将这三者结合起来，形成对上下文的完整理解。
    3.  **优先使用知识库**: 如果知识库中提供了与问题直接相关的信息，请优先基于这些信息进行回答，因为它们是用户指定的权威资料。
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
        try:
            os.makedirs(config.DB_PATH, exist_ok=True)
            self.db_client = chromadb.PersistentClient(path=self.config.DB_PATH)
            self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config.EMBEDDING_MODEL
            )
            self.chat_collection = self.db_client.get_or_create_collection(name=self.config.CHAT_COLLECTION_NAME, embedding_function=self.embedding_func)
            self.fact_memory_collection = self.db_client.get_or_create_collection(name=self.config.FACT_MEMORY_COLLECTION_NAME, embedding_function=self.embedding_func)
            self.event_memory_collection = self.db_client.get_or_create_collection(name=self.config.EVENT_MEMORY_COLLECTION_NAME, embedding_function=self.embedding_func) # V3.0: 初始化事件集合
            self.rag_collection = self.db_client.get_or_create_collection(name=self.config.RAG_COLLECTION_NAME, embedding_function=self.embedding_func)
            logging.info(f"数据库初始化成功: {config.DB_PATH}")
        except Exception as e:
            logging.error(f"初始化 ChromaDB 失败: {e}")
            st.error(f"数据库初始化失败: {e}")
            raise

    def add_document_to_rag(self, user_id: str, file_name: str, file_content: str, progress_callback=None):
        if progress_callback: progress_callback(0, "步骤 1/2: 正在分割文件...")
        chunks = simple_text_splitter(file_content)
        if not chunks:
            if progress_callback: progress_callback(100, "文件内容为空，已跳过。")
            return

        total_chunks = len(chunks)
        logging.info(f"文件 '{file_name}' 被分割成 {total_chunks} 个片段。")
        if progress_callback: progress_callback(5, f"步骤 2/2: 分割完成，准备计算向量... (共 {total_chunks} 块)")
        
        batch_size = 32
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_ids = [f"rag_{user_id}_{file_name}_{time.time()}_{i+j}" for j in range(len(batch_chunks))]
            batch_metadatas = [{"user_id": user_id, "source": file_name} for _ in batch_chunks]
            
            self.rag_collection.add(ids=batch_ids, documents=batch_chunks, metadatas=batch_metadatas)
            
            if progress_callback:
                processed_count = i + len(batch_chunks)
                percentage = min(int((processed_count / total_chunks) * 100), 100)
                status_text = f"步骤 2/2: 正在计算向量... ({processed_count}/{total_chunks})"
                progress_callback(percentage, status_text)
        
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
            
    # V3.0: 重构此函数以同时处理事实和事件
    def save_structured_memory(self, user_id: str, memory_data: dict):
        """保存结构化的记忆，包括静态事实和动态事件"""
        # 1. 保存静态事实
        facts = memory_data.get('static_facts', {})
        if facts and isinstance(facts, dict):
            logging.info(f"正在为用户 {user_id} 保存或更新 {len(facts)} 条事实记忆...")
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
                    metadatas=[{"user_id": user_id, "key": key, "timestamp": time.time()}]
                )

        # 2. 保存动态事件
        events = memory_data.get('events', [])
        if events and isinstance(events, list):
            logging.info(f"正在为用户 {user_id} 保存 {len(events)} 条事件记忆...")
            for event in events:
                if isinstance(event, dict) and 'description' in event:
                    description = event['description']
                    event_time_str = event.get('event_time', '未知时间')
                    doc_id = f"event_{user_id}_{time.time()}"
                    self.event_memory_collection.add(
                        ids=[doc_id],
                        documents=[f"事件：{description}，发生时间：{event_time_str}"],
                        metadatas={"user_id": user_id, "event_time": event_time_str, "saved_at": time.time()}
                    )

    def load_fact_memory(self, user_id: str, top_k: int = 20) -> str:
        """加载静态事实记忆"""
        results = self.fact_memory_collection.get(where={"user_id": user_id}, limit=top_k)
        return "\n".join(f"- {doc}" for doc in results.get('documents', [])) or "暂无"
    
    # V3.0: 新增函数，用于加载和智能筛选事件记忆
    def load_event_memory(self, user_id: str, query: str = None, top_k_similar: int = 3, past_k_recent: int = 5) -> str:
        """加载与用户相关的事件记忆，包括未来的、最近发生的和与查询相关的"""
        if not user_id: return "暂无"
        
        all_events = self.event_memory_collection.get(where={"user_id": user_id})
        if not all_events['ids']: return "暂无"

        # 简单的未来事件识别 (实际应用中可能需要更复杂的日期解析)
        future_events = [doc for doc in all_events['documents'] if any(kw in doc for kw in ["明天", "下周", "将要", "计划"])]
        
        # 获取最近发生的事件
        sorted_events = sorted(zip(all_events['documents'], all_events['metadatas']), key=lambda x: x[1]['saved_at'], reverse=True)
        recent_past_events = [doc for doc, meta in sorted_events[:past_k_recent]]

        # 获取与当前查询最相关的事件
        similar_events = []
        if query:
            query_results = self.event_memory_collection.query(query_texts=[query], where={"user_id": user_id}, n_results=top_k_similar)
            similar_events = query_results.get("documents", [[]])[0]

        # 合并并去重
        final_events = []
        for event_list in [future_events, recent_past_events, similar_events]:
            for event in event_list:
                if event not in final_events:
                    final_events.append(event)

        return "\n".join(f"- {event}" for event in final_events) or "暂无"

    def get_all_event_memory_for_display(self, user_id: str) -> str:
        """获取所有事件记忆用于UI展示"""
        results = self.event_memory_collection.get(where={"user_id": user_id})
        if not results['ids']: return "暂无事件记忆。"
        
        sorted_events = sorted(zip(results['documents'], results['metadatas']), key=lambda x: x[1]['saved_at'], reverse=True)
        return "\n".join(doc for doc, meta in sorted_events)


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
        formatted_docs = "\n".join([f"- {doc}" for doc in retrieved_docs])
        return f"从你的知识库中找到以下相关信息：\n{formatted_docs}"
        
    @st.cache_data(show_spinner=False)
    def query_recent_discussions(_self, user_id: str, query: str, top_k: int = 3) -> str:
        if not _self.chat_collection.get(where={"user_id": user_id}, limit=1)['ids']: return "该用户没有任何历史对话记录。"
        results = _self.chat_collection.query(query_texts=[query], where={"user_id": user_id}, n_results=top_k)
        retrieved_docs = results.get("documents", [[]])[0]
        if not retrieved_docs: return "在你的历史记录中，没有找到与当前问题相关的内容。"
        formatted_docs = "\n".join([f"- \"{doc}\"" for doc in retrieved_docs])
        return f"你回忆起了以下可能相关的历史对话内容：\n{formatted_docs}"

# --- 智能代理类 ---
class ChatAgent:
    def __init__(self, config: Config, db_manager: ChatHistoryDB, api_key: str, user_id: str):
        self.config, self.db_manager, self.api_key, self.user_id = config, db_manager, api_key, user_id
        if not api_key: raise ValueError("必须提供 ZHIPU_API_KEY。")
        self.client = OpenAI(api_key=api_key, base_url=self.config.ZHIPU_BASE_URL)
        self.refresh_agent_state()

    def refresh_agent_state(self, query: str = None):
        """刷新代理状态，加载所有类型的记忆"""
        fact_memory = self.db_manager.load_fact_memory(self.user_id)
        event_memory = self.db_manager.load_event_memory(self.user_id, query=query) # V3.0: 加载事件记忆
        
        system_prompt = self.config.SYSTEM_PROMPT_TEMPLATE.format(
            user_id=self.user_id, 
            long_term_memory=fact_memory,
            event_memory=event_memory
        )
        
        self.messages = self.db_manager.load_history_by_user(self.user_id)
        if not self.messages or self.messages[0]['role'] != 'system':
            self.messages.insert(0, {"role": "system", "content": system_prompt})
        else: 
            self.messages[0]['content'] = system_prompt

    def run(self, user_input: str):
        # V3.0: 在运行前，根据用户输入刷新一次记忆状态，以获取最相关的事件
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

    # V3.0: 核心函数升级，提取事实和事件
    def extract_and_save_memory(self):
        conversation = [msg for msg in self.messages if msg['role'] in ['user', 'assistant']]
        if len(conversation) < 2: return False
        
        full_chat_content = "\n".join([f"{m['role']}: {m['content']}" for m in conversation])
        
        memory_prompt = f"""
        请仔细阅读用户 {self.user_id} 的对话，并以JSON格式，提炼出两种信息：
        1.  `static_facts`: 关于用户的【核心事实】、【长期偏好】或【自定义状态】。这些信息是相对稳定的。例如：姓名、职业、爱好、喜欢的颜色、角色扮演状态如“炼气期”、特定目标等。如果新信息与旧信息冲突，请只保留最新的。
        2.  `events`: 对话中提到的【动态事件】或【未来计划】。每个事件应包含`description`（事件描述）和`event_time`（预估的发生时间，如'2024-08-15 10:00'或'下周三'）。

        如果对话中没有发现任何此类信息，请返回一个空的JSON对象 {{}}。

        对话内容:
        ---
        {full_chat_content}
        ---
        提取的JSON格式示例:
        {{
          "static_facts": {{
            "姓名": "张三",
            "职业": "软件工程师",
            "宠物": "一只名叫'旺财'的狗"
          }},
          "events": [
            {{
              "description": "下周要去北京出差",
              "event_time": "下周"
            }},
            {{
              "description": "完成了项目A的报告",
              "event_time": "昨天"
            }}
          ]
        }}
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
                self.refresh_agent_state() # 保存后立即刷新，确保下一轮对话生效
                return True
            return False
        except Exception as e:
            logging.error(f"解析或保存长期记忆失败: {e}")
            return False

# -----------------------------------------------------------------------------
# 步骤 3: Streamlit 前端界面 (V3.1 - 增加缓存清理工具)
# -----------------------------------------------------------------------------

st.set_page_config(page_title="您的专属记忆助理 V3.1", page_icon="🧠", layout="centered")

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

# --- 侧边栏 ---
with st.sidebar:
    st.header("👤 用户中心")
    user_id_input = st.text_input("请输入您的用户ID", key="user_id_input", placeholder="例如: zhangsan")
    if st.button("登录 / 切换用户", key="login_button"):
        if user_id_input:
            if st.session_state.agent and st.session_state.logged_in_user_id != user_id_input:
                with st.spinner("沉淀最终记忆..."): st.session_state.agent.extract_and_save_memory()
            st.session_state.logged_in_user_id = user_id_input
            st.session_state.agent = None
            st.toast(f"欢迎回来, {user_id_input}！", icon="✅")
            st.rerun()
        else: st.warning("请输入一个用户ID。")

    if st.session_state.logged_in_user_id:
        current_user_id = st.session_state.logged_in_user_id
        st.markdown("---")
        
        st.header("📚 知识库 (RAG)")
        uploaded_file = st.file_uploader("上传知识文件 (.txt/.md)", type=['txt', 'md'], key=f"uploader_{current_user_id}")
        
        if uploaded_file:
            progress_bar = st.progress(0, text="准备上传和学习文件...")
            def update_progress(percent, message):
                progress_bar.progress(percent, text=message)
            try:
                content = uploaded_file.getvalue().decode("utf-8")
                db_manager.add_document_to_rag(current_user_id, uploaded_file.name, content, progress_callback=update_progress)
                time.sleep(2)
                st.toast(f"文件 '{uploaded_file.name}' 已学习完成！", icon="✅")
                st.rerun()
            except Exception as e:
                progress_bar.empty()
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
        
        # V3.1: 新增缓存清理按钮，用于开发和调试
        if st.button("🔄 清理应用缓存", key="clear_app_cache", help="当应用行为异常或代码更新后未生效时，可尝试清理缓存。"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.session_state.agent = None # 强制重新初始化agent
            st.toast("应用缓存已清理！应用将重新加载。", icon="♻️")
            time.sleep(1) # 短暂延迟以确保用户能看到提示
            st.rerun()

        if st.button("🧠 主动提炼记忆", key="extract_memory"):
            if st.session_state.agent:
                status_placeholder = st.empty()
                with st.spinner("⏳ 正在分析和沉淀记忆..."):
                    saved = st.session_state.agent.extract_and_save_memory()
                if saved:
                    status_placeholder.success("✅ 记忆已更新！")
                else:
                    status_placeholder.info("🤷‍ 未发现新的可记忆信息。")
                time.sleep(2)
                status_placeholder.empty()
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
        
        # V3.0: 拆分记忆展示
        with st.expander("👀 查看我的静态事实"):
            memory_content = db_manager.load_fact_memory(current_user_id)
            st.code(memory_content, language=None) if memory_content.strip() and memory_content != "暂无" else st.info("暂无静态事实记忆。")

        with st.expander("📅 查看我的事件记忆"):
            event_memory_content = db_manager.get_all_event_memory_for_display(current_user_id)
            st.code(event_memory_content, language=None) if event_memory_content.strip() and "暂无" not in event_memory_content else st.info("暂无事件记忆。")

    else: st.caption("请先登录以使用全部功能。")

# --- 主聊天界面 ---
st.title("🧠 您的专属记忆助理 V3.1")
if not st.session_state.logged_in_user_id:
    st.info("👈 请在左侧边栏输入用户ID并登录。")
    st.stop()
try:
    if st.session_state.agent is None:
        st.session_state.agent = ChatAgent(config, db_manager, api_key, st.session_state.logged_in_user_id)
except Exception as e:
    st.error(f"初始化助理出错: {e}")
    st.stop()

# 仅显示用户和助手的消息
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
