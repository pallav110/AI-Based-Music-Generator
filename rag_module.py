import os
from typing import Dict, List, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import json
from pathlib import Path
import logging
import hashlib
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("rag_system.log"), logging.StreamHandler()]
)
logger = logging.getLogger("RAGSystem")

class RAGSystem:
    def __init__(self, config: Dict):
        """
        Initialize the RAG system with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.embedding_model = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.knowledge_base = []
        self.embeddings = None
        self.feedback_store = []
        
        # Create feedback directory if it doesn't exist
        os.makedirs(self.config.get('feedback_dir', 'feedback'), exist_ok=True)
        
        # Initialize models
        self._initialize_models()
        
        # Load knowledge base
        self._load_knowledge_base()
        
        # Load feedback data if available
        self._load_feedback()
        
        logger.info(f"RAG System initialized with {len(self.knowledge_base)} documents")
    
    def _initialize_models(self):
        """Initialize embedding and generation models"""
        logger.info("Initializing models...")
        
        # Initialize embedding model
        # Using a more powerful embedding model for better retrieval
        embedding_model_name = self.config.get('embedding_model', 'all-mpnet-base-v2')
        logger.info(f"Loading embedding model: {embedding_model_name}")
        
        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Initialize LLM model and tokenizer
        # Using a larger T5 model for better generation
        model_name = self.config.get('llm_model', 'google/flan-t5-xl')
        logger.info(f"Loading LLM model: {model_name}")
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Move model to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        if device == 'cuda':
            self.llm_model = self.llm_model.cuda()
    
    def _load_knowledge_base(self):
        """Load the knowledge base from file"""
        kb_path = self.config.get('knowledge_base_path', 'knowledge_base.json')
        logger.info(f"Loading knowledge base from {kb_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(kb_path):
                logger.warning(f"Knowledge base file not found at {kb_path}")
                self.knowledge_base = []
                self.embeddings = np.array([])
                return
                
            # Load knowledge base
            with open(kb_path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
                
            # Generate embeddings for all documents
            texts = [item['text'] for item in self.knowledge_base]
            logger.info(f"Generating embeddings for {len(texts)} documents")
            
            self.embeddings = self.embedding_model.encode(
                texts, 
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=32  # Adjust based on available memory
            )
            
            # Convert to numpy array if on CPU
            if not torch.cuda.is_available():
                self.embeddings = self.embeddings.numpy()
                
            logger.info(f"Knowledge base loaded with {len(self.knowledge_base)} entries")
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            self.knowledge_base = []
            self.embeddings = np.array([])
    
    def _load_feedback(self):
        """Load feedback data for reinforcement learning"""
        feedback_path = os.path.join(
            self.config.get('feedback_dir', 'feedback'),
            'feedback_data.json'
        )
        
        if os.path.exists(feedback_path):
            try:
                with open(feedback_path, 'r', encoding='utf-8') as f:
                    self.feedback_store = json.load(f)
                logger.info(f"Loaded {len(self.feedback_store)} feedback entries")
            except Exception as e:
                logger.error(f"Error loading feedback data: {e}")
                self.feedback_store = []
        else:
            logger.info("No feedback data found")
            self.feedback_store = []
    
    def _save_feedback(self):
        """Save feedback data to file"""
        feedback_path = os.path.join(
            self.config.get('feedback_dir', 'feedback'),
            'feedback_data.json'
        )
        
        try:
            with open(feedback_path, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_store, f, indent=2)
            logger.info(f"Saved {len(self.feedback_store)} feedback entries")
        except Exception as e:
            logger.error(f"Error saving feedback data: {e}")
    
    def _retrieve_candidates(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve initial candidate documents from knowledge base.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with scores
        """
        if not self.knowledge_base:
            return []
            
        # Encode query
        query_embedding = self.embedding_model.encode(
            query, 
            convert_to_tensor=True
        )
        
        # Calculate similarity scores
        if torch.cuda.is_available():
            scores = cosine_similarity(
                query_embedding.unsqueeze(0).cpu().numpy(),
                self.embeddings.cpu().numpy()
            )
        else:
            scores = cosine_similarity(
                query_embedding.reshape(1, -1),
                self.embeddings
            )
        
        # Get top_k indices
        top_indices = np.argsort(scores[0])[-top_k:][::-1]
        
        # Return relevant documents with scores
        return [
            {
                "document": self.knowledge_base[i],
                "score": float(scores[0][i]),
                "index": i
            } 
            for i in top_indices
        ]
    
    def _retrieve_relevant_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve and rerank relevant documents from knowledge base.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        # Get more initial candidates than needed
        candidates = self._retrieve_candidates(query, top_k=top_k*2)
        
        if not candidates:
            return []
        
        # Apply reranking based on feedback history if available
        if self.feedback_store:
            candidates = self._rerank_with_feedback(query, candidates)
        
        # Return top documents after potential reranking
        return [item["document"] for item in candidates[:top_k]]
    
    def _rerank_with_feedback(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Rerank candidates based on feedback history.
        
        Args:
            query: User query
            candidates: List of candidate documents with scores
            
        Returns:
            Reranked list of candidates
        """
        # Simple reranking based on historical feedback scores
        query_terms = set(query.lower().split())
        
        for i, candidate in enumerate(candidates):
            boost = 0.0
            doc_id = candidate["document"].get("id", str(candidate["index"]))
            
            # Find relevant feedback entries
            for feedback in self.feedback_store:
                if feedback["document_id"] == doc_id and feedback["rating"] > 0:
                    # Check if the feedback query shares terms with current query
                    feedback_terms = set(feedback["query"].lower().split())
                    term_overlap = len(query_terms.intersection(feedback_terms))
                    
                    if term_overlap > 0:
                        # Boost based on rating and term overlap
                        boost += (feedback["rating"] / 5.0) * (term_overlap / len(query_terms))
            
            # Apply boost to score
            candidates[i]["score"] += boost * 0.2  # Adjust weight as needed
        
        # Re-sort based on updated scores
        return sorted(candidates, key=lambda x: x["score"], reverse=True)
    
    def store_feedback(self, query: str, response: str, documents: List[Dict], rating: int):
        """
        Store user feedback for reinforcement learning.
        
        Args:
            query: User query
            response: Generated response
            documents: Retrieved documents
            rating: User rating (1-5)
        """
        if not documents:
            return
            
        # Create feedback entry
        for i, doc in enumerate(documents):
            doc_id = doc.get("id", str(i))
            
            feedback_entry = {
                "query": query,
                "document_id": doc_id,
                "rating": rating,
                "timestamp": datetime.now().isoformat(),
                "response_hash": hashlib.md5(response.encode()).hexdigest()
            }
            
            self.feedback_store.append(feedback_entry)
        
        # Save updated feedback
        self._save_feedback()
        logger.info(f"Stored feedback for query: {query[:30]}...")
    
    def _create_prompt(self, user_message: str, context_str: str, project_context: Optional[Dict] = None) -> str:
        prompt = (
            f"You are a helpful music production assistant for an AI Song Generator. "
            f"Be friendly, creative, and provide detailed responses to user questions. "
            f"Always help users with lyrics, music production advice, and using the system features.\n\n"
            f"Available features: music generation, lyrics writing, chord progression suggestions, AI singing voice synthesis.\n\n"
            f"Reference information:\n{context_str}\n\n"
            f"{'Project context: ' + str(project_context) if project_context else ''}\n\n"
            f"User: {user_message}\n\n"
            f"Assistant:"
        )
        return prompt
        
    def generate_response(
        self, 
        user_message: str, 
        project_context: Optional[Dict] = None,
        max_length: int = 256,
        temperature: float = 0.7,
        top_k: int = 3
    ) -> Dict:
        """
        Generate a response using RAG approach.
        
        Args:
            user_message: User's message/query
            project_context: Optional context dictionary
            max_length: Maximum length of generated response
            temperature: Temperature for generation
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary containing answer and optional action
        """
        # Track start time for performance logging
        start_time = time.time()
        
        # Retrieve relevant documents
        relevant_docs = self._retrieve_relevant_documents(user_message, top_k=top_k)
        retrieval_time = time.time() - start_time
        
        # Create context string
        context_str = "\n\n".join([doc['text'] for doc in relevant_docs])
        
        # Create prompt for LLM
        prompt = self._create_prompt(user_message, context_str, project_context)
        
        # Generate response
        input_ids = self.llm_tokenizer.encode(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        # Generate output
        generation_start = time.time()
        
        output = self.llm_model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=1
        )
        
        generation_time = time.time() - generation_start
        
        # Decode output
        answer = self.llm_tokenizer.decode(
            output[0], 
            skip_special_tokens=True
        )
        
        # Determine if any action should be taken
        action = self._parse_actions(user_message, answer)
        
        # Generate response ID for feedback
        response_id = hashlib.md5(f"{user_message}:{answer}:{time.time()}".encode()).hexdigest()
        
        # Log performance metrics
        total_time = time.time() - start_time
        logger.info(f"Generated response in {total_time:.2f}s (retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)")
        
        return {
            "answer": answer.strip(),
            "action": action,
            "relevant_documents": relevant_docs,
            "response_id": response_id,
            "performance": {
                "total_time": total_time,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time
            }
        }
    
    def _parse_actions(self, user_message: str, answer: str) -> Optional[Dict]:
        """
        Parse potential actions from user message and generated answer.
        
        Args:
            user_message: User's message/query
            answer: Generated answer
            
        Returns:
            Action dictionary or None
        """
        # Initialize with None
        action = None
        
        # Music-related actions
        if any(term in user_message.lower() for term in ["play", "listen", "music", "song", "audio"]):
            action = "play_music"
        elif any(term in user_message.lower() for term in ["generate", "create", "compose", "make"]):
            if "lyrics" in user_message.lower():
                action = "generate_lyrics"
            else:
                action = "generate_content"
        elif "tempo" in user_message.lower() and any(char.isdigit() for char in user_message):
            # Extract tempo value
            import re
            tempo_matches = re.findall(r'\b(\d+)\s*(?:bpm)?\b', user_message)
            if tempo_matches:
                return {
                    "type": "set_tempo",
                    "value": int(tempo_matches[0])
                }
        
        # Look for specific instructions in the answer
        if "change the genre" in answer.lower() or "select a genre" in answer.lower():
            return {
                "type": "suggest_genre_change"
            }
        
        return action
    
    def add_to_knowledge_base(self, new_data: Dict):
        """
        Add new data to the knowledge base.
        
        Args:
            new_data: Dictionary containing 'text' and optional metadata
        """
        if not isinstance(new_data, dict) or 'text' not in new_data:
            raise ValueError("New data must be a dictionary containing 'text' key")
        
        # Generate a unique ID if not provided
        if "id" not in new_data:
            new_data["id"] = hashlib.md5(new_data["text"].encode()).hexdigest()[:10]
        
        # Add timestamp if not provided
        if "timestamp" not in new_data:
            new_data["timestamp"] = datetime.now().isoformat()
            
        # Add to knowledge base
        self.knowledge_base.append(new_data)
        
        # Generate embedding for new data
        new_embedding = self.embedding_model.encode(
            new_data['text'],
            convert_to_tensor=True
        )
        
        # Add to embeddings
        if torch.cuda.is_available():
            if self.embeddings is None or len(self.embeddings) == 0:
                self.embeddings = new_embedding.unsqueeze(0)
            else:
                self.embeddings = torch.cat([
                    self.embeddings, 
                    new_embedding.unsqueeze(0)
                ])
        else:
            if self.embeddings is None or len(self.embeddings) == 0:
                self.embeddings = new_embedding.unsqueeze(0).numpy()
            else:
                self.embeddings = np.vstack([
                    self.embeddings, 
                    new_embedding.unsqueeze(0).numpy()
                ])
        
        # Save updated knowledge base
        self._save_knowledge_base()
        logger.info(f"Added new document to knowledge base: {new_data.get('id')}")
    
    def _save_knowledge_base(self):
        """Save the knowledge base to file"""
        kb_path = self.config.get('knowledge_base_path', 'knowledge_base.json')
        
        try:
            with open(kb_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2)
            logger.info(f"Saved knowledge base with {len(self.knowledge_base)} entries")
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
    
    def batch_add_documents(self, documents: List[Dict]):
        """
        Add multiple documents to the knowledge base efficiently.
        
        Args:
            documents: List of document dictionaries
        """
        if not documents:
            return
            
        # Process documents in batch
        texts = []
        for doc in documents:
            if 'text' not in doc:
                logger.warning(f"Skipping document without 'text' field")
                continue
                
            # Generate ID if not provided
            if "id" not in doc:
                doc["id"] = hashlib.md5(doc["text"].encode()).hexdigest()[:10]
            
            # Add timestamp
            if "timestamp" not in doc:
                doc["timestamp"] = datetime.now().isoformat()
                
            texts.append(doc["text"])
            self.knowledge_base.append(doc)
        
        # Generate embeddings for all new documents in one batch
        if texts:
            new_embeddings = self.embedding_model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=32
            )
            
            # Add to existing embeddings
            if torch.cuda.is_available():
                if self.embeddings is None or len(self.embeddings) == 0:
                    self.embeddings = new_embeddings
                else:
                    self.embeddings = torch.cat([self.embeddings, new_embeddings])
            else:
                if self.embeddings is None or len(self.embeddings) == 0:
                    self.embeddings = new_embeddings.numpy()
                else:
                    self.embeddings = np.vstack([self.embeddings, new_embeddings.numpy()])
        
        # Save updated knowledge base
        self._save_knowledge_base()
        logger.info(f"Added {len(texts)} new documents to knowledge base")
    
    def index_website_content(self, content_list: List[Dict]):
        """
        Index website content for the knowledge base.
        
        Args:
            content_list: List of content dictionaries with text and metadata
        """
        documents = []
        
        for item in content_list:
            doc = {
                "text": item["text"],
                "metadata": {
                    "source": item.get("source", "website"),
                    "type": item.get("type", "website_content"),
                    "url": item.get("url", ""),
                    "title": item.get("title", "")
                },
                "id": hashlib.md5(f"{item.get('source', '')}:{item['text'][:100]}".encode()).hexdigest()[:10]
            }
            documents.append(doc)
        
        # Add documents in batch
        self.batch_add_documents(documents)
    
    @classmethod
    def from_default_config(cls):
        """Create a RAGSystem with enhanced default configuration"""
        config = {
            'embedding_model': 'all-mpnet-base-v2',  # Better semantic understanding
            'llm_model': 'google/flan-t5-large',     # More powerful generation
            'knowledge_base_path': 'knowledge_base.json',
            'feedback_dir': 'feedback'
        }
        return cls(config)