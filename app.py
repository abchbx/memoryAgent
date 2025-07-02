# -*- coding: utf-8 -*-
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
import hashlib
import re

# -----------------------------------------------------------------------------
# 步骤 2: 后端逻辑代码
# -----------------------------------------------------------------------------

# --- 日志记录配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 核心配置类 ---
class Config:
    """集中管理所有配置"""
    load_dotenv()
    ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
    LLM_MODEL = "glm-4-flash"
    THINKING_MODEL = "glm-z1-flash"
    EMBEDDING_MODEL = "<填入向量模型地址>"
    DB_PATH = "<数据存储路径>" 
    CHAT_COLLECTION_NAME = "user_chat_history"
    FACT_MEMORY_COLLECTION_NAME = "user_fact_memory"
    EVENT_MEMORY_COLLECTION_NAME = "user_event_memory"
    RAG_COLLECTION_NAME = "user_rag_documents"
    TIMEZONE = "Asia/Shanghai"
    SYSTEM_PROMPT_TEMPLATE = """
    你是一位为用户 {user_id} 服务的、充满温度与支持的私人伙伴。

    # 你的核心角色:
    - **成为伙伴, 而非老师**: 你的首要目标是成为一个乐于助人、有同理心的伙伴。你的角色不是纠正用户的错误，而是陪伴和支持他们梳理思绪、规划生活。
    - **积极、鼓励、有温度**: 始终保持积极和鼓励的态度。在回应时，多一些关心和理解，少一些生硬的指令和说教。
    - **个性化互动**: 像一个真正的朋友一样，自然地运用我为你提供的所有信息（用户的个人事实、事件记忆、知识库等），以便更好地理解上下文，并给出贴心、个性化的回应。

    # 参考信息 (我会为你提供):
    
    ## 1. 时间与日期
    - **当前精确时间**: {current_time}
    - **未来一周日期参考**: 
      为了帮助你准确计算日期，这里是接下来一周的日期信息。**请优先使用此信息回答与日期相关的问题。**
      {date_reference}

    ## 2. 关于用户的记忆
    - **长期事实**: {long_term_memory}
    - **未来计划**: {future_events}
    - **近期事件**: {past_events}

    # 你的回应方式:
    - 深入理解用户的意图，结合所有已知信息，生成一个自然、流畅、且充满伙伴感的回答。
    - 如果知识库信息相关，请以一种建议或“我发现这个可能有用”的口吻来分享，而不是作为绝对事实。
    """

def process_response_for_display(content: str) -> str:
    """使用正则表达式移除<think>...</think>标签及其中的所有内容。"""
    return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

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
            if meta.get('is_permanent') is not True and meta.get('timestamp') and meta['timestamp'] < today_start_timestamp
        ]

        if ids_to_delete:
            logging.info(f"为用户 {user_id} 找到 {len(ids_to_delete)} 条旧的临时事实记忆，准备删除...")
            self.fact_memory_collection.delete(ids=ids_to_delete)
            logging.info(f"已成功为用户 {user_id} 清理了 {len(ids_to_delete)} 条旧的临时事实记忆。")
        else:
            logging.info(f"用户 {user_id} 没有今天之前的旧临时事实记忆可清理。")

    def clear_expired_events(self, user_id: str):
        logging.info(f"开始为用户 {user_id} 清理过期的事件...")
        
        now = datetime.datetime.now(self.tz)
        
        results = self.event_memory_collection.get(where={"user_id": user_id})
        if not results['ids']:
            logging.info(f"用户 {user_id} 没有事件记忆可清理。")
            return

        ids_to_delete = []
        for i, meta in enumerate(results['metadatas']):
            event_time_iso = meta.get('event_time_iso')
            if not event_time_iso:
                continue
            
            try:
                parsed_dt = datetime.datetime.fromisoformat(event_time_iso.replace('Z', '+00:00'))
                if parsed_dt.tzinfo is None:
                    event_dt = self.tz.localize(parsed_dt)
                else:
                    event_dt = parsed_dt.astimezone(self.tz)
                
                if event_dt < now:
                    ids_to_delete.append(results['ids'][i])
            except (ValueError, TypeError):
                logging.warning(f"无法解析用户 {user_id} 的事件时间戳: {event_time_iso}，跳过清理。")
                continue

        if ids_to_delete:
            logging.info(f"为用户 {user_id} 找到 {len(ids_to_delete)} 条过期事件，准备删除...")
            self.event_memory_collection.delete(ids=ids_to_delete)
            logging.info(f"已成功为用户 {user_id} 清理了 {len(ids_to_delete)} 条过期事件。")
        else:
            logging.info(f"用户 {user_id} 没有过期的事件可清理。")
            
    def save_structured_memory(self, user_id: str, memory_data: dict):
        """保存结构化的记忆，处理事实和智能事件操作"""

        def _save_facts(facts: dict, is_permanent: bool):
            if not (facts and isinstance(facts, dict)): return
            
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
                    metadatas=[{"user_id": user_id, "key": key, "timestamp": time.time(), "is_permanent": is_permanent}]
                )

        permanent_facts = memory_data.get('permanent_facts', {})
        _save_facts(permanent_facts, is_permanent=True)

        temporal_facts = memory_data.get('temporal_facts', {})
        if not temporal_facts and 'static_facts' in memory_data:
             temporal_facts = memory_data.get('static_facts', {})
        _save_facts(temporal_facts, is_permanent=False)

        event_actions = memory_data.get('event_actions', [])
        if event_actions and isinstance(event_actions, list):
            logging.info(f"正在为用户 {user_id} 执行 {len(event_actions)} 条事件操作...")
            
            ids_to_upsert, docs_to_upsert, metas_to_upsert = [], [], []
            ids_to_delete = []

            for action in event_actions:
                action_type = action.get('type')
                event_id = action.get('event_id')
                if not (action_type and event_id): continue

                db_id = f"event_{user_id}_{event_id}"

                if action_type == 'upsert':
                    data = action.get('data', {})
                    description = data.get('description')
                    event_time_iso = data.get('event_time_iso')
                    if description and event_time_iso:
                        ids_to_upsert.append(db_id)
                        docs_to_upsert.append(description)
                        metas_to_upsert.append({
                            "user_id": user_id,
                            "event_id": event_id,
                            "event_time_iso": event_time_iso,
                            "saved_at": time.time()
                        })
                elif action_type == 'delete':
                    ids_to_delete.append(db_id)

            if ids_to_upsert:
                self.event_memory_collection.upsert(ids=ids_to_upsert, documents=docs_to_upsert, metadatas=metas_to_upsert)
                logging.info(f"已为用户 {user_id} 新增/更新 {len(ids_to_upsert)} 条事件。")
            
            if ids_to_delete:
                self.event_memory_collection.delete(ids=ids_to_delete)
                logging.info(f"已为用户 {user_id} 删除 {len(ids_to_delete)} 条事件。")

    def load_fact_memory(self, user_id: str, top_k: int = 20) -> str:
        results = self.fact_memory_collection.get(where={"user_id": user_id}, limit=top_k)
        return "\n".join(f"- {doc}" for doc in results.get('documents', [])) or "暂无"
    
    def load_event_memory(self, user_id: str, query: str = None) -> tuple[str, str]:
        if not user_id: return "暂无", "暂无"
        
        all_events_result = self.event_memory_collection.get(where={"user_id": user_id})
        if not all_events_result['ids']: return "暂无", "暂无"

        now = datetime.datetime.now(self.tz)
        future_events, past_events = [], []

        for i, meta in enumerate(all_events_result['metadatas']):
            event_time_iso = meta.get('event_time_iso')
            if not event_time_iso: continue
            
            try:
                parsed_dt = datetime.datetime.fromisoformat(event_time_iso.replace('Z', '+00:00'))
                if parsed_dt.tzinfo is None:
                    event_dt = self.tz.localize(parsed_dt)
                else:
                    event_dt = parsed_dt
            except (ValueError, TypeError):
                continue

            event_record = (event_dt, all_events_result['documents'][i])
            if event_dt > now:
                future_events.append(event_record)
            else:
                past_events.append(event_record)

        future_events.sort(key=lambda x: x[0])
        past_events.sort(key=lambda x: x[0], reverse=True)

        future_str = "\n".join(f"- {doc} (时间: {dt.astimezone(self.tz).strftime('%Y-%m-%d %H:%M')})" for dt, doc in future_events) or "暂无"
        past_str = "\n".join(f"- {doc} (时间: {dt.astimezone(self.tz).strftime('%Y-%m-%d %H:%M')})" for dt, doc in past_events[:5]) or "暂无"

        return future_str, past_str

    def get_all_event_memory_for_display(self, user_id: str) -> str:
        results = self.event_memory_collection.get(where={"user_id": user_id})
        if not results['ids']: return "暂无事件记忆。"
        
        events_with_time = []
        aware_min_dt = datetime.datetime.min.replace(tzinfo=pytz.utc)

        for i, meta in enumerate(results['metadatas']):
            event_time_iso = meta.get('event_time_iso')
            doc = results['documents'][i]
            dt = aware_min_dt
            
            if event_time_iso:
                try:
                    parsed_dt = datetime.datetime.fromisoformat(event_time_iso.replace('Z', '+00:00'))
                    if parsed_dt.tzinfo is None:
                        dt = self.tz.localize(parsed_dt)
                    else:
                        dt = parsed_dt
                except (ValueError, TypeError):
                    doc = f"{doc} (时间解析失败: {event_time_iso})"
            
            events_with_time.append((dt, doc))
        
        events_with_time.sort(key=lambda x: x[0], reverse=True)
        
        formatted_events = []
        for dt, doc in events_with_time:
            if dt == aware_min_dt:
                formatted_events.append(f"- {doc}")
            else:
                local_dt = dt.astimezone(self.tz)
                formatted_events.append(f"- {doc} (时间: {local_dt.strftime('%Y-%m-%d %H:%M')})")

        return "\n".join(formatted_events) or "暂无事件记忆。"

    def get_all_events_for_llm(self, user_id: str) -> list[dict]:
        results = self.event_memory_collection.get(where={"user_id": user_id})
        if not results['ids']: return []
        
        events = []
        for i, meta in enumerate(results['metadatas']):
            if 'event_id' in meta and 'event_time_iso' in meta:
                events.append({
                    "event_id": meta['event_id'],
                    "description": results['documents'][i],
                    "event_time_iso": meta['event_time_iso']
                })
        return events

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

    def _get_date_reference(self, now: datetime.datetime) -> str:
        """创建未来一周的日期参考字符串"""
        weekdays_zh = {"Monday": "星期一", "Tuesday": "星期二", "Wednesday": "星期三", 
                       "Thursday": "星期四", "Friday": "星期五", "Saturday": "星期六", "Sunday": "星期日"}
        date_references = []
        for i in range(7):
            future_date = now + datetime.timedelta(days=i)
            day_name = ""
            if i == 0: day_name = " (今天)"
            elif i == 1: day_name = " (明天)"
            elif i == 2: day_name = " (后天)"
            
            weekday_en = future_date.strftime('%A')
            weekday_zh = weekdays_zh.get(weekday_en, weekday_en)
            
            date_references.append(
                f"  - {future_date.strftime('%Y-%m-%d')}{day_name}, {weekday_zh}"
            )
        return "\n".join(date_references)

    def refresh_agent_state(self, query: str = None):
        """刷新代理状态时，注入精确的日期参考信息"""
        self.fact_memory_str = self.db_manager.load_fact_memory(self.user_id)
        self.future_events_str, self.past_events_str = self.db_manager.load_event_memory(self.user_id, query=query)
        self.all_events_display_str = self.db_manager.get_all_event_memory_for_display(self.user_id)
        
        now_time = datetime.datetime.now(self.tz)
        current_time_str = now_time.strftime('%Y-%m-%d %H:%M:%S %Z')
        
        date_reference_str = self._get_date_reference(now_time)

        system_prompt = self.config.SYSTEM_PROMPT_TEMPLATE.format(
            user_id=self.user_id,
            current_time=current_time_str,
            date_reference=date_reference_str,
            long_term_memory=self.fact_memory_str,
            future_events=self.future_events_str,
            past_events=self.past_events_str
        )
        
        self.messages = self.db_manager.load_history_by_user(self.user_id)
        if not self.messages or self.messages[0]['role'] != 'system':
            self.messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            self.messages[0]['content'] = system_prompt

    def run(self, user_input: str):
        """
        【V5.6 修改】: 重构状态管理，确保在所有写入操作后才刷新状态。
        """
        # 1. 刷新状态，为当前对话准备上下文
        self.refresh_agent_state(query=user_input)
        
        # 2. 准备发送给 LLM 的消息列表
        context_from_rag = self.db_manager.query_rag_documents(self.user_id, user_input)
        context_from_history = self.db_manager.query_recent_discussions(self.user_id, user_input)
        
        messages_for_llm = list(self.messages)
        if context_from_rag: messages_for_llm.append({"role": "system", "content": f"补充信息(来自你的知识库，可以参考一下):\n{context_from_rag}"})
        if context_from_history: messages_for_llm.append({"role": "system", "content": f"补充信息(来自我们的历史对话，也许能派上用场):\n{context_from_history}"})
        
        messages_for_llm.append({"role": "user", "content": user_input})
        
        use_thinking_model = st.session_state.get('use_thinking_model', False)
        model_to_use = self.config.THINKING_MODEL if use_thinking_model else self.config.LLM_MODEL
        logging.info(f"正在使用模型: {model_to_use}")

        # 3. 调用 LLM 并流式生成响应
        full_response_content = ""
        try:
            stream = self.client.chat.completions.create(
                model=model_to_use, 
                messages=messages_for_llm,
                stream=True
            )
            for chunk in stream:
                content_chunk = chunk.choices[0].delta.content or ""
                full_response_content += content_chunk
                yield content_chunk
                
        except Exception as e:
            logging.error(f"调用LLM API失败: {e}")
            full_response_content = f"抱歉，出错了: {e}"
            yield full_response_content
        
        # --- 流式输出结束后，执行后续的写入和状态更新 ---
        
        # 4. 更新内存中的对话历史
        user_message = {"role": "user", "content": user_input}
        assistant_message = {"role": "assistant", "content": full_response_content}
        self.messages.extend([user_message, assistant_message])
        
        # 5. 将新对话写入数据库
        self.db_manager.save_message(self.user_id, user_message)
        self.db_manager.save_message(self.user_id, assistant_message)
        
        # 6. 提取并写入新的记忆到数据库
        self.extract_and_save_memory()
        
        # 7. 在所有写入操作完成后，最后统一刷新一次状态，确保下一轮对话的上下文是完整的
        self.refresh_agent_state()

    def extract_and_save_memory(self):
        """【V5.6 修改】: 移除此函数中的状态刷新，交由主流程 run() 统一管理"""
        conversation = [msg for msg in self.messages if msg['role'] in ['user', 'assistant']]
        if len(conversation) < 2: return False
        
        recent_conversation = conversation[-6:]
        full_chat_content = "\n".join([f"{m['role']}: {m['content']}" for m in recent_conversation])
        
        now_time = datetime.datetime.now(self.tz)
        current_time_iso = now_time.isoformat()

        existing_events = self.db_manager.get_all_events_for_llm(self.user_id)
        existing_events_str = json.dumps(existing_events, ensure_ascii=False, indent=2) if existing_events else "[]"

        date_reference_str = self._get_date_reference(now_time)
        memory_prompt = f"""
        你现在是一位为用户 {self.user_id} 服务的、高效的记忆管家。你的核心任务是分析最新的对话，并管理用户的个人信息和日程事件。

        # 1. 已有信息参考
        - **当前精确时间**: `{current_time_iso}`
        - **未来一周日期参考 (用于精确解析时间)**:
{date_reference_str}
        - **已记录的日程事件**: 
        ```json
        {existing_events_str}
        ```

        # 2. 你的任务
        请仔细阅读下面的最新对话，并以一个JSON对象的格式，总结出你需要执行的操作。这个JSON对象应包含两部分：`permanent_facts` 和 `event_actions`。

        ## `permanent_facts` (用户的核心事实)
        - 提炼关于用户的、几乎不会改变的核心信息（如姓名、职业、长期偏好）。
        - 格式为键值对。如果没有，则为空对象 `{{}}`。

        ## `event_actions` (日程事件管理)
        - 这是一个操作指令的列表，用于管理日程。
        - **分析对话**：判断对话是在**创建新事件**、**更新现有事件**还是**删除现有事件**。
        - **参考已有事件**：利用上面提供的“已记录的日程事件”列表来判断一个事件是新的还是已存在的。
        - **生成操作指令**：根据你的判断，生成一个或多个操作指令。每个指令都是一个JSON对象，包含 `type`, `event_id`, 和 `data` (仅upsert需要)。

        ### 操作指令详解:
        
        1.  **创建/更新事件 (`upsert`)**:
            - `type`: "upsert"
            - `event_id`: **(关键!)**
                - 如果是**新事件**，请根据事件核心内容（如“公司会议”、“生日派对”）和日期创造一个简短、唯一的英文ID，例如 `evt_meeting_20250703`。
                - 如果是**更新事件**，请从“已记录的日程事件”列表中找到并使用**完全相同**的 `event_id`。
            - `data`:
                - `description`: 事件的完整描述。
                - `event_time_iso`: **必须**将对话中的时间（如“明天下午3点”）**严格参照上面提供的日期参考**，解析为标准的ISO 8601格式 (`YYYY-MM-DDTHH:MM:SS±HH:MM`)。

        2.  **删除事件 (`delete`)**:
            - `type`: "delete"
            - `event_id`: **(关键!)** 从“已记录的日程事件”列表中找到用户想要删除的事件，并使用其 `event_id`。

        # 3. 输出格式要求
        - 最终输出必须是一个完整的、可被解析的JSON对象。
        - 如果没有提取到任何信息，则每个字段对应的值应为空对象或空列表。
        - 示例: `{{"permanent_facts": {{"nickname": "小明"}}, "event_actions": [{{"type": "upsert", "event_id": "evt_go_to_mars_20250801", "data": {{"description": "用户计划去火星", "event_time_iso": "2025-08-01T09:00:00+08:00"}}}}, {{"type": "delete", "event_id": "evt_old_meeting_20250701"}}]}}`

        ---
        最新对话内容:
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
                has_new_data = (extracted_data.get('permanent_facts') or 
                                extracted_data.get('event_actions'))
                if has_new_data:
                    self.db_manager.save_structured_memory(self.user_id, extracted_data)
                    logging.info(f"用户 {self.user_id} 的新记忆已处理。")
                    return True
            return False
        except Exception as e:
            logging.error(f"解析或保存长期记忆失败: {e}\n响应内容: {content if 'content' in locals() else 'N/A'}")
            return False

# -----------------------------------------------------------------------------
# 步骤 3: Streamlit 前端界面
# -----------------------------------------------------------------------------

st.set_page_config(page_title="您的专属记忆伙伴", page_icon="🤗", layout="centered")

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
if 'use_thinking_model' not in st.session_state: st.session_state.use_thinking_model = False


# --- 侧边栏 ---
with st.sidebar:
    st.header("👤 用户中心")
    user_id_input = st.text_input("请输入您的用户ID", key="user_id_input", placeholder="例如: zhangsan")
    if st.button("登录 / 切换用户", key="login_button"):
        if user_id_input:
            if st.session_state.agent and st.session_state.logged_in_user_id != user_id_input:
                st.session_state.agent.extract_and_save_memory()
            
            st.session_state.logged_in_user_id = user_id_input
            
            with st.spinner(f"正在为您清理过期的临时记忆..."):
                db_manager.clear_old_temporal_fact_memory(user_id_input)
                time.sleep(0.5)

            with st.spinner(f"正在为您清理过期的日程..."):
                db_manager.clear_expired_events(user_id_input)
                time.sleep(0.5)

            st.session_state.agent = None
            st.toast(f"欢迎回来, {user_id_input}！很高兴再次见到你。", icon="🤗")
            st.rerun()
        else: st.warning("请输入一个用户ID。")

    if st.session_state.logged_in_user_id:
        current_user_id = st.session_state.logged_in_user_id
        st.markdown("---")
        
        st.header("⚙️ 模型设置")
        use_thinking_model_toggle = st.toggle(
            f"启用高级思考模型 ({config.THINKING_MODEL})", 
            key='use_thinking_model', 
            help=f"开启后，对话将使用更强大的 {config.THINKING_MODEL} 模型。默认使用 {config.LLM_MODEL}。"
        )

        st.markdown("---")
        
        st.header("📚 知识库 (RAG)")
        uploaded_file = st.file_uploader("分享一些资料给我学习 (.txt/.md)", type=['txt', 'md'], key=f"uploader_{current_user_id}")
        
        if uploaded_file is not None and uploaded_file.file_id != st.session_state.get('last_uploaded_file_id'):
            progress_container = st.empty()
            try:
                def update_progress(percent, message):
                    progress_container.progress(percent, text=message)
                
                content = uploaded_file.getvalue().decode("utf-8")
                db_manager.add_document_to_rag(current_user_id, uploaded_file.name, content, progress_callback=update_progress)
                
                st.session_state.last_uploaded_file_id = uploaded_file.file_id
                
                time.sleep(1)
                progress_container.empty()
                st.toast(f"谢谢你的分享，'{uploaded_file.name}' 我已经学习完啦！", icon="✅")
                st.rerun()
            except Exception as e:
                st.session_state.last_uploaded_file_id = uploaded_file.file_id
                progress_container.empty()
                st.error(f"处理文件失败: {e}")
        
        rag_files = db_manager.get_rag_file_list(current_user_id)
        if rag_files:
            with st.expander("查看我学习过的资料", expanded=False):
                for f in rag_files: st.caption(f)
            if st.button("🗑️ 忘记所有学习资料", key="clear_rag"):
                with st.spinner("正在清空我的学习笔记..."):
                    db_manager.clear_user_rag_documents(current_user_id)
                    st.cache_data.clear()
                st.toast("我已经忘记所有学习过的资料啦。", icon="🗑️")
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

        if st.button("🧠 手动回顾一下", key="extract_memory", help="通常我会自动记忆，这个按钮可以让我立即强制回顾我们的对话。"):
            if st.session_state.agent:
                with st.spinner("正在强制回顾对话..."):
                    saved = st.session_state.agent.extract_and_save_memory()
                    # 在手动回顾后也刷新一次主状态
                    st.session_state.agent.refresh_agent_state()
                if saved:
                    st.success("✅ 回顾完成，又有新收获！")
                else:
                    st.info("🤷‍ 暂时没有发现新的信息可以记录。")
                time.sleep(2)
                st.rerun()

        if st.button("🧹 开始一段新对话", key="clear_chat"):
            if st.session_state.agent:
                st.session_state.agent.extract_and_save_memory()
                with st.spinner("正在清空我们的对话..."):
                    db_manager.clear_user_chat_history(current_user_id)
                    st.cache_data.clear()
                st.session_state.agent = None
                st.toast("好了，我们可以开始新的话题了！", icon="💬")
                st.rerun()
        
        with st.expander("👀 看看关于你的记忆"):
            if 'agent' in st.session_state and st.session_state.agent:
                memory_content = st.session_state.agent.fact_memory_str
                if memory_content.strip() and memory_content != "暂无":
                    st.code(memory_content, language=None)
                else:
                    st.info("关于你的事，我还了解得不多。")

        with st.expander("📅 看看我们的日程和事件"):
            if 'agent' in st.session_state and st.session_state.agent:
                event_memory_content = st.session_state.agent.all_events_display_str
                if event_memory_content.strip() and "暂无" not in event_memory_content:
                    st.code(event_memory_content, language=None)
                else:
                    st.info("我们之间还没有发生什么特别的事。")

    else: st.caption("请先登录，让我认识你。")

# --- 主聊天界面 ---
st.title("🤗 您的专属记忆伙伴 V5.6")
st.caption("我在这里，随时准备倾听、支持和陪伴。")
if not st.session_state.logged_in_user_id:
    st.info("👈 请在左侧边栏输入你的ID，让我认识你吧。")
    st.stop()
try:
    if st.session_state.agent is None:
        st.session_state.agent = ChatAgent(config, db_manager, api_key, st.session_state.logged_in_user_id)
except Exception as e:
    st.error(f"初始化伙伴时出错了: {e}")
    st.stop()

for message in st.session_state.agent.messages:
    if message["role"] in ["user", "assistant"]:
        with st.chat_message(message["role"]):
            content_to_display = message["content"]
            if message["role"] == "assistant":
                content_to_display = process_response_for_display(content_to_display)
            
            if content_to_display:
                st.markdown(content_to_display)

if prompt := st.chat_input(f"嗨, {st.session_state.logged_in_user_id}, 在想些什么呢?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for chunk in st.session_state.agent.run(prompt):
            full_response += chunk
            placeholder.markdown(process_response_for_display(full_response) + "▌")
        placeholder.markdown(process_response_for_display(full_response))
