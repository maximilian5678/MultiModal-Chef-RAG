import sqlite3
import faiss
import numpy as np
import re
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import config
import json

class RecipeRetriever:
    def __init__(self, title_index_path, recipe_ids_path):
        self.title_index_path = title_index_path
        self.recipe_ids_path = recipe_ids_path
    
    def _load_faiss_titles(self) -> faiss.Index:
        return faiss.read_index(self.title_index_path)
    
    def _load_id_mapping(self) -> np.ndarray:
        return np.load(self.recipe_ids_path)
    
    def _embed_query(self, query: str):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        query_embedding = model.encode(query, convert_to_numpy=True).astype("float32")
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        return query_embedding
    
    def similarity_search(self, query: str, topk: int) -> List[Dict]:
        user_query_embedding = self._embed_query(query)
        scores, indicies = self._load_faiss_titles().search(user_query_embedding, topk)

        results = []
        for idx, score in zip(indicies[0], scores[0]): # Find based on the index the recipe in the SQLite db
            if idx == -1:
                continue
            recipe_id = self._load_id_mapping()[idx]
            recipe = self._fetch_recipe_by_id(recipe_id)
            if recipe is not None:
                recipe['similarity_score'] = float(score) # Add similarity score to the recipe dict
                results.append(recipe)
                
        if len(results) == 0:
            results = self._keyword_fallback(query, topk)
        return results

    def _fetch_recipe_by_id(self, recipe_id: int) -> Optional[Dict]:
        connection = sqlite3.connect(config.DB_PATH)
        cursor = connection.cursor()
        row = cursor.execute(
            "SELECT title, ingredients, directions FROM recipes WHERE recipe_id = ?",
            (recipe_id,)
        ).fetchone()

        if row is None:
            return None
        
        return {
            "recipe_id": recipe_id,
            "title": row[0],
            "ingredients": row[1],
            "directions": row[2]
        }
    
    def _keyword_fallback(self, query: str, top_k: int) -> List[Dict]:
        connection = sqlite3.connect(config.DB_PATH)
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT recipe_id, title, ingredients, directions
            FROM recipes
            WHERE title LIKE ? OR ingredients LIKE ?
            LIMIT ?
            """,
            (f"%{query}%", f"%{query}%", top_k),
        )

        results = []
        for row in cursor.fetchall():
            results.append({
                "recipe_id": row[0],
                "title": row[1],
                "ingredients": row[2],
                "directions": row[3],
                "similarity_score": None,
            })    
        return results

    def format_as_context(self,recipes: List[Dict]) -> str:
        """Format retrieved recipes into a structured context string for the chatbot."""
        context_blocks = []

        for i, recipe in enumerate(recipes, start=1):
            ingredients = json.loads(recipe["ingredients"]) # Json string to list
            directions = json.loads(recipe["directions"])

            block = [
                f"### Recipe {i}",
                f"Title: {recipe['title']}",
                "",
                "Ingredients:",
                *[f"- {ing}" for ing in ingredients],
                "",
                "Instructions:",
                *[f"{j+1}. {step}" for j, step in enumerate(directions)],
            ]

            context_blocks.append("\n".join(block))

        return "\n\n---\n\n".join(context_blocks)
    
    def get_recipe_instructions(self, generated_response: str, recipes: List[Dict]) -> List[Dict]:
        """Extract instruction steps from the chatbot's generated response."""
        steps = []

        for line in generated_response.splitlines():
            line = line.strip()
            # Match lines that start with a number followed by a period and space
            match = re.match(r"^(\d+)\.\s+(.*)", line)
            if match:
                step_id = int(match.group(1)) # Extract the step number
                instruction = match.group(2) # Extract the instruction text

                steps.append({
                    "step_id": step_id,
                    "instruction": instruction
                })
        
        # Fallback if chatbot did not return any instruction steps
        # Use instructions directly from the retrieved recipe
        if not steps:
            directions = json.loads(recipes[0]["directions"]) # Json string to list

            for i, direction in enumerate(directions, start=1):
                steps.append({
                    "step_id": i,
                    "instruction": direction.strip()
                })
        
        return steps

    
if __name__ == "__main__":
    retriever = RecipeRetriever(
        title_index_path= config.FAISS_TITLE_INDEX_PATH,
        recipe_ids_path=config.RECIPE_IDS_PATH
    )

    recipes = retriever.similarity_search("How do i make a chocolate cake?", config.TOP_K_TITLE)
    context = retriever.format_as_context(recipes)
    print(context)