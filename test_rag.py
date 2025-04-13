# debug_test_rag.py
import json
import sys
import traceback
from rag_module import RAGSystem

def inspect_object(obj, name="Object"):
    """Print detailed information about an object"""
    print(f"\n--- {name} Inspection ---")
    print(f"Type: {type(obj)}")
    if hasattr(obj, '__dict__'):
        print(f"Attributes: {obj.__dict__.keys()}")
    
    if isinstance(obj, list):
        print(f"Length: {len(obj)}")
        if len(obj) > 0:
            print(f"First item type: {type(obj[0])}")
            if isinstance(obj[0], dict):
                print(f"First item keys: {obj[0].keys()}")
    elif isinstance(obj, dict):
        print(f"Keys: {obj.keys()}")

def main():
    try:
        # Initialize the RAG system with debug output
        print("Initializing RAG system...")
        rag_system = RAGSystem.from_default_config()
        
        # Examine the initial state
        print("\nInitial RAG system state:")
        print(f"Knowledge base: {rag_system.knowledge_base}")
        print(f"Embeddings shape: {rag_system.embeddings.shape if hasattr(rag_system.embeddings, 'shape') else 'N/A'}")
        
        # Create a single test item first
        print("\nTesting with a single knowledge item first...")
        test_item = {
            "text": "Music theory is the study of music practices.",
            "source": "test_source"
        }
        
        # Add the item and verify
        rag_system.add_to_knowledge_base(test_item)
        print(f"Knowledge base after adding one item: {len(rag_system.knowledge_base)} items")
        
        # Test a simple query with the single item
        print("\nTesting query with single knowledge item...")
        test_query = "What is music theory?"
        
        response = rag_system.generate_response(
            user_message=test_query,
            project_context=None  # Simplify by not using context initially
        )
        
        print(f"Query: {test_query}")
        print(f"Answer: {response['answer']}")
        print(f"Action: {response['action']}")
        
        # Inspect the relevant documents
        print("\nInspecting relevant documents:")
        inspect_object(response['relevant_documents'], "Relevant Documents")
        
        if response['relevant_documents']:
            for i, doc in enumerate(response['relevant_documents']):
                print(f"\nDocument {i+1}:")
                for key, value in doc.items():
                    print(f"  {key}: {value}")
        
        print("\nBasic test completed successfully!")
        
        # Continue with multiple items if basic test passes
        print("\n--- Continuing with multiple knowledge items ---")
        
        # Add some more test knowledge
        more_test_knowledge = [
            {
                "text": "A chord is a group of notes played together.",
                "source": "chord_theory"
            },
            {
                "text": "Song structure includes intro, verse, chorus, bridge, and outro.",
                "source": "songwriting_guide"
            }
        ]
        
        # Add knowledge to the system
        for item in more_test_knowledge:
            rag_system.add_to_knowledge_base(item)
            print(f"Added: {item['source']}")
        
        # Test another query with multiple knowledge items
        print("\nTesting another query with multiple knowledge items...")
        test_query = "How do I structure a song?"
        
        response = rag_system.generate_response(
            user_message=test_query,
            project_context=None
        )
        
        print(f"Query: {test_query}")
        print(f"Answer: {response['answer']}")
        print(f"Action: {response['action']}")
        
        # Inspect the relevant documents (safely)
        print("\nRelevant documents:")
        if response['relevant_documents']:
            for i, doc in enumerate(response['relevant_documents']):
                print(f"\nDocument {i+1}:")
                for key, value in doc.items():
                    print(f"  {key}: {value}")
        else:
            print("No relevant documents found")
        
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc(file=sys.stdout)

if __name__ == "__main__":
    main()