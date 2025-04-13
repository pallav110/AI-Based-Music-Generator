import os
from typing import Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
from pathlib import Path

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
        self.knowledge_base = None
        self.embeddings = None
        
        # Initialize models
        self._initialize_models()
        
        # Load knowledge base
        self._load_knowledge_base()
    
    def _initialize_models(self):
        """Initialize embedding and generation models"""
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            self.config.get('embedding_model', 'all-MiniLM-L6-v2'),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Initialize LLM model and tokenizer
        model_name = self.config.get('llm_model', 'google/flan-t5-base')
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.llm_model = self.llm_model.cuda()
    
    def _load_knowledge_base(self):
        """Load the knowledge base from file"""
        kb_path = self.config.get('knowledge_base_path', 'knowledge_base.json')
        
        try:
            # Check if file exists
            if not os.path.exists(kb_path):
                print(f"Warning: Knowledge base file not found at {kb_path}")
                self.knowledge_base = []
                self.embeddings = np.array([])
                return
                
            # Load knowledge base
            with open(kb_path, 'r') as f:
                self.knowledge_base = json.load(f)
                
            # Generate embeddings for all documents
            texts = [item['text'] for item in self.knowledge_base]
            self.embeddings = self.embedding_model.encode(
                texts, 
                convert_to_tensor=True,
                show_progress_bar=True
            )
            
            # Convert to numpy array if on CPU
            if not torch.cuda.is_available():
                self.embeddings = self.embeddings.numpy()
                
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            self.knowledge_base = []
            self.embeddings = np.array([])
    
    def _retrieve_relevant_documents(self, query: str, top_k: int = 3) -> list:
        """
        Retrieve relevant documents from knowledge base.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
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
                query_embedding.unsqueeze(0).cpu(),
                self.embeddings.cpu()
            )
        else:
            scores = cosine_similarity(
                query_embedding.reshape(1, -1),
                self.embeddings
            )
        
        # Get top_k indices
        top_indices = np.argsort(scores[0])[-top_k:][::-1]
        
        # Return relevant documents
        return [self.knowledge_base[i] for i in top_indices]
    
    def generate_response(
        self, 
        user_message: str, 
        project_context: Optional[Dict] = None,
        max_length: int = 200,
        temperature: float = 0.7
    ) -> Dict:
        """
        Generate a response using RAG approach.
        
        Args:
            user_message: User's message/query
            project_context: Optional context dictionary
            max_length: Maximum length of generated response
            temperature: Temperature for generation
            
        Returns:
            Dictionary containing answer and optional action
        """
        # Retrieve relevant documents
        relevant_docs = self._retrieve_relevant_documents(user_message)
        
        # Create context string
        context_str = "\n\n".join([doc['text'] for doc in relevant_docs])
        
        # Add project context if provided
        if project_context:
            context_str += f"\n\nProject Context:\n{json.dumps(project_context, indent=2)}"
        
        # Create prompt for LLM
        prompt = (
            f"Use the following context to answer the user's question. "
            f"If you don't know the answer, say you don't know.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {user_message}\n\n"
            f"Answer:"
        )
        
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
        output = self.llm_model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=1
        )
        
        # Decode output
        answer = self.llm_tokenizer.decode(
            output[0], 
            skip_special_tokens=True
        )
        
        # Determine if any action should be taken
        action = None
        if "play" in user_message.lower() or "music" in user_message.lower():
            action = "play_music"
        elif "generate" in user_message.lower() or "create" in user_message.lower():
            action = "generate_content"
        
        return {
            "answer": answer,
            "action": action,
            "relevant_documents": relevant_docs
        }
    
    def add_to_knowledge_base(self, new_data: Dict):
        """
        Add new data to the knowledge base.
        
        Args:
            new_data: Dictionary containing 'text' and optional metadata
        """
        if not isinstance(new_data, dict) or 'text' not in new_data:
            raise ValueError("New data must be a dictionary containing 'text' key")
            
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
    
    def _save_knowledge_base(self):
        """Save the knowledge base to file"""
        kb_path = self.config.get('knowledge_base_path', 'knowledge_base.json')
        
        try:
            with open(kb_path, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)
        except Exception as e:
            print(f"Error saving knowledge base: {e}")
    
    @classmethod
    def from_default_config(cls):
        """Create a RAGSystem with default configuration"""
        config = {
            'embedding_model': 'all-MiniLM-L6-v2',
            'llm_model': 'google/flan-t5-base',
            'knowledge_base_path': 'knowledge_base.json'
        }
        return cls(config)