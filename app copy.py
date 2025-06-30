# -----------------------------------------------------------------------------
# 步骤 1: 导入所有必要的库
# -----------------------------------------------------------------------------
import os
import json
import logging
import time
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# -----------------------------------------------------------------------------
# 步骤 2: 后端逻辑代码 (V2.6 - RAG缓存优化版)
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
    MEMORY_COLLECTION_NAME = "user_entity_memory"
    RAG_COLLECTION_NAME = "user_rag_documents"

    SYSTEM_PROMPT_TEMPLATE = """
    你是为用户 {user_id} 服务的顶级个人智能助手，拥有卓越的记忆、推理和知识库查询能力。
    # 关于用户 {user_id} 的已知信息 (你的长期记忆):
    {long_term_memory}
    # 你的工作流程:
    1.  **深入理解**: 分析用户的最新问题。
    2.  **结合记忆与知识**: 我会为你提供三类信息：用户的长期记忆、相关的历史对话、以及从用户上传的知识库中检索到的相关资料。你必须将这三者结合起来，形成对上下文的完整理解。
    3.  **优先使用知识库**: 如果知识库中提供了与问题直接相关的信息，请优先基于这些信息进行回答，因为它们是用户指定的权威资料。
    4.  **个性化回答**: 基于所有信息，为用户 {user_id} 生成一个富有洞察力、连贯且个性化的回答。
    """
    
    MEMORY_EXTRACTION_INTERVAL = 4

# --- 文本分割器 ---
def simple_text_splitter(text: str, max_chunk_size: int = 500) -> list[str]:
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
            self.memory_collection = self.db_client.get_or_create_collection(name=self.config.MEMORY_COLLECTION_NAME, embedding_function=self.embedding_func)
            self.rag_collection = self.db_client.get_or_create_collection(name=self.config.RAG_COLLECTION_NAME, embedding_function=self.embedding_func)
            logging.info(f"数据库初始化成功: {config.DB_PATH}")
        except Exception as e:
            logging.error(f"初始化 ChromaDB 失败: {e}")
            st.error(f"数据库初始化失败: {e}")
            raise

    # ... 其他数据库方法保持不变 ...
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
            
    def save_entities_to_memory(self, user_id: str, entities: dict):
        if not entities: return
        for key, value in entities.items():
            doc_id = f"memory_{user_id}_{key}"
            self.memory_collection.upsert(ids=[doc_id], documents=[f"事实: {key} 是 {value}。"], metadatas=[{"user_id": user_id, "key": key, "timestamp": time.time()}])

    def load_long_term_memory(self, user_id: str, top_k: int = 20) -> str:
        results = self.memory_collection.get(where={"user_id": user_id}, limit=top_k)
        return "\n".join(results.get('documents', [])) or "暂无"
        
    def add_document_to_rag(self, user_id: str, file_name: str, file_content: str):
        chunks = simple_text_splitter(file_content)
        if not chunks: return
        doc_ids = [f"rag_{user_id}_{file_name}_{time.time()}_{i}" for i in range(len(chunks))]
        metadatas = [{"user_id": user_id, "source": file_name} for _ in chunks]
        self.rag_collection.add(ids=doc_ids, documents=chunks, metadatas=metadatas)

    def clear_user_rag_documents(self, user_id: str):
        if self.rag_collection.get(where={"user_id": user_id})['ids']:
            self.rag_collection.delete(where={"user_id": user_id})
            logging.info(f"已清空用户 {user_id} 的知识库。")

    def get_rag_file_list(self, user_id: str) -> list[str]:
        results = self.rag_collection.get(where={"user_id": user_id})
        if not results['ids']: return []
        return sorted(list({meta['source'] for meta in results['metadatas'] if 'source' in meta}))

    # ⭐ V2.6 核心优化: 为RAG检索函数增加缓存
    # 这将缓存相同用户对相同查询的检索结果，避免重复计算
    @st.cache_data(show_spinner=False)
    def query_rag_documents(_self, user_id: str, query: str, top_k: int = 3) -> str:
        """从用户专属的RAG知识库中检索信息 (带缓存)"""
        logging.info(f"正在为用户 {user_id} 执行RAG检索，查询: '{query}'")
        if not _self.rag_collection.get(where={"user_id": user_id}, limit=1)['ids']:
            return ""

        results = _self.rag_collection.query(query_texts=[query], where={"user_id": user_id}, n_results=top_k)
        retrieved_docs = results.get("documents", [[]])[0]
        if not retrieved_docs: return ""
        
        formatted_docs = "\n".join([f"- {doc}" for doc in retrieved_docs])
        return f"从你的知识库中找到以下相关信息：\n{formatted_docs}"
        
    @st.cache_data(show_spinner=False)
    def query_recent_discussions(_self, user_id: str, query: str, top_k: int = 3) -> str:
        """从历史对话中检索信息 (带缓存)"""
        logging.info(f"正在为用户 {user_id} 执行历史对话检索，查询: '{query}'")
        if not _self.chat_collection.get(where={"user_id": user_id}, limit=1)['ids']:
            return "该用户没有任何历史对话记录。"
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

    def refresh_agent_state(self):
        long_term_memory = self.db_manager.load_long_term_memory(self.user_id)
        system_prompt = self.config.SYSTEM_PROMPT_TEMPLATE.format(user_id=self.user_id, long_term_memory=long_term_memory)
        self.messages = self.db_manager.load_history_by_user(self.user_id)
        if not self.messages or self.messages[0]['role'] != 'system':
            self.messages.insert(0, {"role": "system", "content": system_prompt})
        else: self.messages[0] = {"role": "system", "content": system_prompt}

    def run(self, user_input: str):
        # 使用缓存的检索函数
        context_from_rag = self.db_manager.query_rag_documents(self.user_id, user_input)
        context_from_history = self.db_manager.query_recent_discussions(self.user_id, user_input)
        
        messages_for_llm = list(self.messages)
        if context_from_rag: messages_for_llm.append({"role": "system", "content": context_from_rag})
        if context_from_history: messages_for_llm.append({"role": "system", "content": context_from_history})
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
            self.extract_and_save_memory(is_periodic=True)
        return final_response

    def extract_and_save_memory(self, is_periodic=False):
        conversation = [msg for msg in self.messages if msg['role'] in ['user', 'assistant']]
        if len(conversation) < 2: return False
        full_chat_content = "\n".join([f"{m['role']}: {m['content']}" for m in conversation])
        memory_prompt = f"""请仔细阅读用户 {self.user_id} 的对话，并以JSON格式，提炼出核心事实和长期偏好。如果没有，返回{{}}。\n对话内容:\n---\n{full_chat_content}\n---\n提取的JSON:"""
        try:
            response = self.client.chat.completions.create(model=self.config.LLM_MODEL, messages=[{"role": "user", "content": memory_prompt}], response_format={"type": "json_object"})
            content = response.choices[0].message.content
            if content and (extracted_data := json.loads(content)):
                self.db_manager.save_entities_to_memory(self.user_id, extracted_data)
                self.refresh_agent_state()
                return True
            return False
        except Exception as e:
            logging.error(f"解析或保存长期记忆失败: {e}")
            return False

# -----------------------------------------------------------------------------
# 步骤 3: Streamlit 前端界面 (V2.6 - 前端逻辑无变化)
# -----------------------------------------------------------------------------

st.set_page_config(page_title="您的专属记忆助理 V2.6", page_icon="🧠", layout="centered")

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
            with st.spinner(f"学习文件: {uploaded_file.name}..."):
                try:
                    content = uploaded_file.getvalue().decode("utf-8")
                    db_manager.add_document_to_rag(current_user_id, uploaded_file.name, content)
                    st.toast(f"文件 '{uploaded_file.name}' 已添加！", icon="✅")
                    st.rerun()
                except Exception as e: st.error(f"处理文件失败: {e}")
        
        rag_files = db_manager.get_rag_file_list(current_user_id)
        if rag_files:
            with st.expander("查看已上传的文件", expanded=True):
                for f in rag_files: st.caption(f)
            if st.button("🗑️ 清空所有知识库文件", key="clear_rag"):
                with st.spinner("清空知识库..."):
                    db_manager.clear_user_rag_documents(current_user_id)
                    st.cache_data.clear() # 清空所有缓存
                st.toast("知识库已清空！", icon="🗑️")
                st.rerun()

        st.markdown("---")
        st.header("🛠️ 记忆工具箱")
        if st.button("🧠 主动提炼记忆", key="extract_memory"):
            if st.session_state.agent:
                with st.spinner("分析和沉淀记忆..."): saved = st.session_state.agent.extract_and_save_memory()
                st.toast("记忆已更新！" if saved else "未发现新的可记忆信息。", icon="�")
                st.rerun()
        if st.button("🧹 清理当前对话", key="clear_chat"):
            if st.session_state.agent:
                with st.spinner("保存最后的回忆..."): st.session_state.agent.extract_and_save_memory()
                with st.spinner("清空对话历史..."):
                    db_manager.clear_user_chat_history(current_user_id)
                    st.cache_data.clear() # 清空所有缓存
                st.session_state.agent = None
                st.toast("对话历史已清空！", icon="🗑️")
                st.rerun()
        with st.expander("👀 查看我的长期记忆"):
            memory_content = db_manager.load_long_term_memory(current_user_id)
            st.code(memory_content, language=None) if memory_content.strip() and memory_content != "暂无" else st.info("暂无长期记忆。")
    else: st.caption("请先登录以使用全部功能。")

# --- 主聊天界面 ---
st.title("🧠 您的专属记忆助理 V2.6")
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
