#!/usr/bin/env python3
"""
Benchmark: Confidence Improvement Methods
Comparing GrACE, Credence (Calibration Game), RENT, and Enhanced Dirichlet+Topology
"""

import numpy as np
import json
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    f1_score, matthews_corrcoef, brier_score_loss, 
    accuracy_score, precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import warnings
from datetime import datetime
from scipy.stats import dirichlet, entropy
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import time
from pathlib import Path
import re
import random
import requests

warnings.filterwarnings('ignore')

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ============================================================================
# TEXT ENCODER WITH FALLBACK
# ============================================================================

class MinimalEncoder:
    
    def __init__(self, max_features=128):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            dtype=np.float32,
            max_df=0.9,
            min_df=2
        )
        self.fitted = False
        self.embedding_dim = max_features
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
    def encode(self, texts):
        if isinstance(texts, str):
            cache_key = hash(texts) % 10000  # Limit hash space
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key].copy()
            self._cache_misses += 1
            texts = [texts]
            single = True
        else:
            single = False
        
        if not self.fitted:
            self.vectorizer.fit(texts)
            self.fitted = True
        
        try:
            embeddings = self.vectorizer.transform(texts).toarray()
            if embeddings.shape[1] < self.embedding_dim:
                padding = np.zeros((embeddings.shape[0], self.embedding_dim - embeddings.shape[1]))
                embeddings = np.concatenate([embeddings, padding], axis=1)
            elif embeddings.shape[1] > self.embedding_dim:
                embeddings = embeddings[:, :self.embedding_dim]
            
            return embeddings[0] if len(texts) == 1 else embeddings
        except:
            return np.random.randn(self.embedding_dim) if len(texts) == 1 else np.random.randn(len(texts), self.embedding_dim)
    
    def clear_cache(self):
        """Manually clear cache to free memory"""
        self._cache.clear()
        print(f"Cache cleared. Hits: {self._cache_hits}, Misses: {self._cache_misses}")
        
_ENCODER = None

def get_encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = MinimalEncoder()
    return _ENCODER

# ============================================================================
# LLAMA MODEL INTERFACE (Replace with actual API)
# ============================================================================

class LlamaModel:
    """Interface to Llama model - REPLACE WITH ACTUAL API"""
    
    def __init__(self, model_name="llama-8b"):
        self.model_name = model_name
        self.base_accuracy = 0.72
        self._response_cache = {}
        print(f"Initialized {model_name} (placeholder - replace with actual API)")
    
    def generate(self, prompt, temperature=0.7, max_tokens=512, return_logits=False):
        """Generate response - REPLACE WITH ACTUAL LLAMA API CALL"""
        # TODO: Replace with actual API
        # response = llama_api.generate(prompt, temperature, max_tokens)
        cache_key = (prompt, temperature)
        if cache_key in self._response_cache:
            if return_logits:
                return self._response_cache[cache_key], np.random.randn(10, 50257).astype(np.float32)
            return self._response_cache[cache_key]
        
        response = self._mock_llama_response(prompt, temperature)
        self._response_cache[cache_key] = response

        if len(self._response_cache) > 500:
            self._response_cache.pop(next(iter(self._response_cache)))
        
        if return_logits:
            # Mock logits for demonstration
            logits = np.random.randn(10, 50257)  # [batch, vocab_size]
            return response, logits
        return response
    
    def _mock_llama_response(self, prompt, temperature):
        """Mock response - REPLACE THIS"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['calculate', 'solve', 'what is']):
            numbers = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', prompt)]
            if len(numbers) >= 2:
                return str(int(sum(numbers)))
            return "42"
        elif any(f'{letter})' in prompt for letter in ['a', 'b', 'c', 'd']):
            return random.choice(['A', 'B', 'C', 'D'])
        elif any(word in prompt_lower for word in ['can', 'do', 'does', 'is', 'are']):
            return random.choice(['yes', 'no'])
        
        return "I need to think step by step."

# ============================================================================
# TOPOLOGY + DIRICHLET (YOUR EXISTING METHOD)
# ============================================================================

class EnhancedTopologyRiskExtractor:
    """Topology extractor from your original code"""
    
    def __init__(self):
        self.min_samples = 2
        self._distance_cache = {}
    
    def extract_risk_features(self, reasoning_embeddings, k_paths=7):
        if len(reasoning_embeddings) < 2:
            return self._get_default_risk_features()
        
        embeddings = np.array(reasoning_embeddings[:k_paths], dtype=np.float32)
        risk_features = {}
        
        distances = self._get_cached_distances(embeddings)
        
        try:
            risk_features['reasoning_spread'] = float(np.std(distances))
            risk_features['consistency_score'] = self._compute_consistency_fast(embeddings)
            risk_features['complexity_entropy'] = self._compute_complexity_fast(embeddings)
            risk_features['stability_score'] = self._compute_stability_fast(embeddings)
            risk_features['coherence_score'] = self._compute_coherence_fast(embeddings)
            risk_features['diversity_penalty'] = float(max(0, (np.mean(distance) - 1.0) * 0.5))
            risk_features['outlier_risk'] = self._compute_outlier_risk_fast(embeddings)
            risk_features['cluster_quality'] = self._compute_cluster_quality(embeddings)
            risk_features['risk_score'] = self._compute_enhanced_risk_score(risk_features)
        except Exception as e:
            return self._get_default_risk_features()
            
        return risk_features

    def _get_cached_distances(self, embeddings):
        """Cache distance calculations"""
        cache_key = embeddings.tobytes()
        if cache_key not in self._distance_cache:
            distance = pdist(embeddings, metric='euclidean')
            self._distance_cache[cache_key] = distances
            if len(self._distance_cache) > 100:
                self._distance_cache.pop(next(iter(self._distance_cache)))
        return self._distance_cache[cache_key]
    
    def _compute_consistency_fast(self, embeddings):
        if len(embeddings) < 2:
            return 0.5
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        similarities = normalized @ normalized.T
        avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
        return float(1 - avg_similarity)
    
    def _compute_complexity_fast(self, distances):
        if len(distances) == 0:
            return 1.0
        complexity = np.std(distances) / (np.mean(distances) + 1e-10)
        return float(np.clip(complexity, 0, 5))
    
    def _compute_stability_fast(self, embeddings):
        try:
            centroid = np.mean(embeddings, axis=0)
            distances = np.linalg.norm(embeddings - centroid, axis = 1)
            coherence_risk = np.std(distances) / (np.mean(distances) + 1e-10)
            return float(np.clip(risk, 0, 2))
        except:
            return 1.0
    
    def _compute_coherence_fast(self, embeddings):
        try:
            centroid = np.mean(embeddings, axis=0)
            distances_to_centroid = [np.linalg.norm(emb - centroid) for emb in embeddings]
            coherence_risk = np.std(distances_to_centroid) / (np.mean(distances_to_centroid) + 1e-10)
            return float(np.clip(coherence_risk, 0, 3))
        except:
            return 1.0
    
    def _compute_diversity_penalty(self, embeddings):
        try:
            pairwise_distances = pdist(embeddings)
            diversity = np.mean(pairwise_distances)
            penalty = max(0, (diversity - 1.0) * 0.5)
            return float(np.clip(penalty, 0, 2))
        except:
            return 0.5
    
    def _compute_outlier_risk_fast(self, embeddings):
        try:
            if len(embeddings) < 3:
                return 0.5
            centroid = np.mean(embeddings, axis=0)
            distances = [np.linalg.norm(emb - centroid) for emb in embeddings]
            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1
            outlier_threshold = q3 + 1.5 * iqr
            n_outliers = sum(1 for d in distances if d > outlier_threshold)
            return float(n_outliers / len(embeddings))
        except:
            return 0.5
    
    def _compute_cluster_quality(self, embeddings):
        try:
            if len(embeddings) < 3:
                return 0.5
            from sklearn.metrics import silhouette_score
            best_score = -1
            for n_clusters in range(2, min(len(embeddings), 5)):
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
                    cluster_labels = kmeans.fit_predict(embeddings)
                    score = silhouette_score(embeddings, cluster_labels)
                    best_score = max(best_score, score)
                except:
                    continue
            cluster_risk = 1 - ((best_score + 1) / 2)
            return float(np.clip(cluster_risk, 0, 1))
        except:
            return 0.5
    
    def _compute_enhanced_risk_score(self, features):
        weights = {
            'reasoning_spread': 0.2,
            'consistency_score': 0.25,
            'complexity_entropy': 0.1,
            'stability_score': 0.2,
            'coherence_score': 0.1,
            'diversity_penalty': 0.05,
            'outlier_risk': 0.05,
            'cluster_quality': 0.05
        }
        risk_score = sum(weights[k] * features[k] for k in weights.keys() if k in features)
        return float(np.clip(risk_score, 0, 3))
    
    def _get_default_risk_features(self):
        return {
            'reasoning_spread': 1.0,
            'consistency_score': 1.0,
            'complexity_entropy': 1.0,
            'stability_score': 1.0,
            'coherence_score': 1.0,
            'diversity_penalty': 0.5,
            'outlier_risk': 0.5,
            'cluster_quality': 0.5,
            'risk_score': 1.0
        }


class DirichletConfidenceHead(nn.Module):
    """Dirichlet confidence head from your original code"""
    
    def __init__(self, embedding_dim, num_classes=2, hidden_dim=128):
        super(DirichletConfidenceHead, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x):
        logits = self.network(x)
        alphas = F.softplus(logits) + 1.0
        concentration = torch.sum(alphas, dim=-1, keepdim=True)
        mean_probs = alphas / concentration
        
        return {
            'alphas': alphas,
            'concentration': concentration,
            'mean_probs': mean_probs,
            'logits': logits
        }


class EnhancedDirichletTopologyRisk:
    """Your existing Dirichlet + Topology method"""
    
    def __init__(self, embedding_dim=384, num_classes=2):
        self.dirichlet_head = DirichletConfidenceHead(embedding_dim, num_classes)
        self.topology_extractor = EnhancedTopologyRiskExtractor()
        self.encoder = get_text_encoder()
        self._train_dirichlet_head()
        self.dirichlet_weight = 0.4
        self.topology_weight = 0.6
    
    def _train_dirichlet_head(self):
        X_train = torch.randn(1000, 384)
        y_train = torch.randint(0, 2, (1000,))
        optimizer = torch.optim.Adam(self.dirichlet_head.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        self.dirichlet_head.train()
        for epoch in range(20):
            optimizer.zero_grad()
            outputs = self.dirichlet_head(X_train)
            loss = criterion(outputs['logits'], y_train)
            loss.backward()
            optimizer.step()
        
        self.dirichlet_head.eval()
    
    def compute_confidence(self, reasoning_embeddings, k_reasoning_paths=7):
        if not reasoning_embeddings:
            return {'confidence': 0.5, 'risk_score': 1.0}
        
        risk_features = self.topology_extractor.extract_risk_features(
            reasoning_embeddings, k_reasoning_paths
        )
        
        embedding_tensor = torch.tensor(reasoning_embeddings[0], dtype=torch.float16).unsqueeze(0)
        
        with torch.no_grad():
            dirichlet_output = self.dirichlet_head(embedding_tensor)
            dirichlet_confidence = self._compute_dirichlet_confidence(dirichlet_output)
        
        topology_confidence = self._compute_topology_confidence(risk_features)
        
        fused_confidence = (
            self.dirichlet_weight * dirichlet_confidence + 
            self.topology_weight * topology_confidence
        )
        
        final_confidence = self._apply_risk_mitigation(fused_confidence, risk_features)
        
        return {
            'confidence': float(np.clip(final_confidence, 0.01, 0.99)),
            'risk_score': risk_features['risk_score'],
            'risk_features': risk_features
        }
    
    def _compute_topology_confidence(self, risk_features):
        base_confidence = 1.0 / (1.0 + risk_features['risk_score'])
        coherence_bonus = max(0, (1.0 - risk_features['coherence_score']) * 0.1)
        diversity_adjustment = -risk_features['diversity_penalty'] * 0.05
        outlier_penalty = -risk_features['outlier_risk'] * 0.1
        cluster_bonus = max(0, (1.0 - risk_features['cluster_quality']) * 0.05)
        
        enhanced_confidence = (base_confidence + coherence_bonus + 
                             diversity_adjustment + outlier_penalty + cluster_bonus)
        return np.clip(enhanced_confidence, 0.01, 0.99)
    
    def _apply_risk_mitigation(self, base_confidence, risk_features):
        if risk_features['risk_score'] > 2.0:
            risk_penalty = min(0.2, (risk_features['risk_score'] - 2.0) * 0.1)
            base_confidence -= risk_penalty
        
        if risk_features['consistency_score'] < 0.3:
            consistency_bonus = (0.3 - risk_features['consistency_score']) * 0.1
            base_confidence += consistency_bonus
        
        if risk_features['stability_score'] < 0.5:
            stability_bonus = (0.5 - risk_features['stability_score']) * 0.05
            base_confidence += stability_bonus
        
        return np.clip(base_confidence, 0.01, 0.99)
    
    def _compute_dirichlet_confidence(self, dirichlet_output):
        alphas = dirichlet_output['alphas'].squeeze()
        concentration = dirichlet_output['concentration'].squeeze()
        max_prob = torch.max(dirichlet_output['mean_probs']).item()
        precision_conf = torch.sigmoid(concentration - len(alphas)).item()
        entropy_conf = 1.0 / (1.0 + torch.sum(torch.digamma(alphas) - torch.digamma(concentration)).item())
        confidence = (max_prob + precision_conf + entropy_conf) / 3.0
        return np.clip(confidence, 0.01, 0.99)

# ============================================================================
# METHOD 1: GrACE (Generative Confidence Estimation)
# Based on: https://arxiv.org/html/2508.14390v1
# ============================================================================

class GrACEMethod:
    """
    GrACE: Uses internal model data to generate a special confidence token
    The model is trained to append <confidence: X.XX> to its answers
    """
    
    def __init__(self, model, name="GrACE"):
        self.model = model
        self.name = name
        self.encoder = get_text_encoder()
        self.confidence_token = "<confidence:"
        
        # In real implementation, you would fine-tune the model to output confidence tokens
        # For now, we simulate this behavior
        print(f"Initialized {name}: Generative confidence token method")
    
    def solve_and_get_confidence(self, problem, dataset_type, k_reasoning_paths=3):
        """
        Generate answer with embedded confidence token
        In practice, the model is trained to output: "Answer: 42 <confidence: 0.85>"
        """
        if dataset_type == 'stock':
            # Stock prediction path
            features = problem['features'].reshape(1, -1)
            probs = self.model.predict_proba(features)[0]
            prediction = np.argmax(probs)
            
            # Simulate confidence token generation based on probability spread
            confidence = self._extract_grace_confidence(probs, features)
            reasoning_embeddings = []
            
            return prediction, confidence, reasoning_embeddings
        else:
            # Text generation path
            prompt = f"Question: {problem['question']}\nAnswer with your confidence level:"
            
            # In real implementation: answer = model.generate(prompt)
            # Model would output: "The answer is 42 <confidence: 0.85>"
            answer = self.model.generate(prompt, temperature=0.3)
            
            # Extract confidence from generated token or compute from internal state
            confidence = self._extract_grace_confidence_from_text(answer, problem)
            
            # Generate reasoning embeddings for later analysis
            reasoning_embeddings = self._generate_reasoning_embeddings(problem, k_reasoning_paths)
            
            return answer, confidence, reasoning_embeddings
    
    def _extract_grace_confidence(self, probs, features):
        """
        Extract confidence from model's internal state
        In real GrACE, this comes from a special token generated by the model
        """
        # Simulate confidence based on probability distribution
        max_prob = np.max(probs)
        entropy_val = entropy(probs)
        
        # GrACE learns to calibrate this during training
        # Higher max prob and lower entropy = higher confidence
        confidence = max_prob * (1 - entropy_val / np.log(len(probs)))
        
        # Add slight randomness to simulate learned calibration
        confidence = confidence * np.random.uniform(0.9, 1.1)
        
        return float(np.clip(confidence, 0.01, 0.99))
    
    def _extract_grace_confidence_from_text(self, answer, problem):
        """Extract confidence token from generated text"""
        # Check if model generated confidence token
        if self.confidence_token in answer:
            try:
                conf_str = answer.split(self.confidence_token)[1].split(">")[0].strip()
                return float(conf_str)
            except:
                pass
        
        # Fallback: estimate confidence from answer characteristics
        answer_length = len(answer)
        has_hedging = any(word in answer.lower() for word in ['maybe', 'probably', 'might', 'unsure'])
        
        base_confidence = 0.7
        if has_hedging:
            base_confidence *= 0.8
        if answer_length < 20:
            base_confidence *= 0.9
        
        return float(np.clip(base_confidence, 0.01, 0.99))
    
    def _generate_reasoning_embeddings(self, problem, k_paths):
        """Generate reasoning embeddings for consistency analysis"""
        embeddings = []
        for i in range(k_paths):
            reasoning_text = f"GrACE reasoning {i+1}: {problem['question']}"
            embeddings.append(self.encoder.encode(reasoning_text))
        return embeddings

# ============================================================================
# METHOD 2: Credence (Calibration Game)
# Based on: https://arxiv.org/html/2505.22660v1
# ============================================================================

class CredenceMethod:
    """
    Credence: Iterative calibration through feedback game
    Model predicts confidence, receives feedback on calibration, reassesses
    """
    
    def __init__(self, model, name="Credence", n_iterations=3):
        self.model = model
        self.name = name
        self.n_iterations = n_iterations
        self.encoder = get_text_encoder()
        self.calibration_history = []
        print(f"Initialized {name}: Calibration game with {n_iterations} iterations")
    
    def solve_and_get_confidence(self, problem, dataset_type, k_reasoning_paths=3):
        """
        Iteratively refine confidence through calibration feedback
        """
        if dataset_type == 'stock':
            features = problem['features'].reshape(1, -1)
            probs = self.model.predict_proba(features)[0]
            prediction = np.argmax(probs)
            
            # Iterative calibration game
            confidence = self._calibration_game_stock(probs, features)
            reasoning_embeddings = []
            
            return prediction, confidence, reasoning_embeddings
        else:
            # Initial answer generation
            prompt = f"Question: {problem['question']}\nProvide your answer and confidence:"
            initial_answer = self.model.generate(prompt, temperature=0.4)
            
            # Calibration game iterations
            confidence, final_answer = self._calibration_game_text(
                initial_answer, problem, k_reasoning_paths
            )
            
            reasoning_embeddings = self._generate_reasoning_embeddings(problem, k_reasoning_paths)
            
            return final_answer, confidence, reasoning_embeddings
    
    def _calibration_game_stock(self, probs, features):
        """Run calibration game for stock predictions"""
        confidence = np.max(probs)
        
        for iteration in range(self.n_iterations):
            # Simulate calibration feedback
            feedback = self._generate_calibration_feedback(confidence, iteration)
            
            # Adjust confidence based on feedback
            if feedback == 'overconfident':
                confidence *= 0.9
            elif feedback == 'underconfident':
                confidence *= 1.05
            
            confidence = np.clip(confidence, 0.01, 0.99)
        
        return float(confidence)
    
    def _calibration_game_text(self, initial_answer, problem, k_paths):
        """
        Run calibration game for text generation
        Iteratively refine confidence through feedback
        """
        current_answer = initial_answer
        confidence = 0.7  # Initial guess
        
        for iteration in range(self.n_iterations):
            # Generate feedback prompt
            feedback_prompt = self._create_feedback_prompt(
                problem, current_answer, confidence, iteration
            )
            
            # Model reassesses based on feedback
            # In real implementation: reassessment = model.generate(feedback_prompt)
            reassessment = self.model.generate(feedback_prompt, temperature=0.3)
            
            # Extract new confidence from reassessment
            new_confidence = self._extract_confidence_from_reassessment(
                reassessment, confidence
            )
            
            # Update confidence
            confidence = new_confidence
            
            # Optionally update answer if model indicates error
            if 'incorrect' in reassessment.lower():
                current_answer = reassessment
        
        return float(np.clip(confidence, 0.01, 0.99)), current_answer
    
    def _generate_calibration_feedback(self, confidence, iteration):
        """Generate calibration feedback for the model"""
        # Simulate feedback based on confidence level
        if confidence > 0.85:
            return 'overconfident'
        elif confidence < 0.55:
            return 'underconfident'
        else:
            return 'well_calibrated'
    
    def _create_feedback_prompt(self, problem, answer, confidence, iteration):
        """Create prompt for calibration feedback iteration"""
        return f"""Question: {problem['question']}
Your previous answer: {answer}
Your confidence: {confidence:.2f}

Iteration {iteration + 1}: Reassess your answer and confidence.
Are you overconfident or underconfident? Should you revise?
New answer and confidence:"""
    
    def _extract_confidence_from_reassessment(self, reassessment, prev_confidence):
        """Extract updated confidence from model's reassessment"""
        # Look for confidence indicators in text
        confidence_words = {
            'certain': 0.95, 'sure': 0.85, 'confident': 0.8,
            'likely': 0.7, 'probably': 0.65, 'maybe': 0.5,
            'unsure': 0.4, 'unlikely': 0.3, 'doubtful': 0.2
        }
        
        reassessment_lower = reassessment.lower()
        for word, conf_val in confidence_words.items():
            if word in reassessment_lower:
                # Weighted average with previous confidence
                return 0.7 * conf_val + 0.3 * prev_confidence
        
        # If no clear indicator, slightly adjust previous confidence
        return prev_confidence * np.random.uniform(0.95, 1.05)
    
    def _generate_reasoning_embeddings(self, problem, k_paths):
        """Generate reasoning embeddings"""
        embeddings = []
        for i in range(k_paths):
            reasoning_text = f"Credence calibration {i+1}: {problem['question']}"
            embeddings.append(self.encoder.encode(reasoning_text))
        return embeddings

# ============================================================================
# METHOD 3: RENT (Reinforcement Learning with Entropy)
# Based on: Reinforcement learning using model entropy
# ============================================================================

class RENTMethod:
    """
    RENT: Reinforcement learning method using entropy to improve reasoning
    Reinforces highly confident (low entropy) chain-of-thought paths
    """
    
    def __init__(self, model, name="RENT", entropy_threshold=0.5):
        self.model = model
        self.name = name
        self.entropy_threshold = entropy_threshold
        self.encoder = get_text_encoder()
        self.reward_history = []
        print(f"Initialized {name}: RL-based confidence refinement (entropy threshold={entropy_threshold})")
    
    def solve_and_get_confidence(self, problem, dataset_type, k_reasoning_paths=5):
        """
        Generate multiple CoT paths, reinforce low-entropy ones
        """
        if dataset_type == 'stock':
            features = problem['features'].reshape(1, -1)
            probs = self.model.predict_proba(features)[0]
            prediction = np.argmax(probs)
            
            # Use entropy-based confidence
            confidence = self._compute_entropy_confidence(probs)
            reasoning_embeddings = []
            
            return prediction, confidence, reasoning_embeddings
        else:
            # Generate multiple reasoning paths with varying temperatures
            reasoning_paths = self._generate_multiple_cot_paths(
                problem, k_reasoning_paths
            )
            
            # Compute entropy for each path
            path_entropies = self._compute_path_entropies(reasoning_paths)
            
            # Select and reinforce low-entropy paths
            selected_answer, confidence = self._reinforcement_selection(
                reasoning_paths, path_entropies
            )
            
            # Generate reasoning embeddings from selected paths
            reasoning_embeddings = self._generate_reasoning_embeddings(
                reasoning_paths, path_entropies
            )
            
            return selected_answer, confidence, reasoning_embeddings
    
    def _generate_multiple_cot_paths(self, problem, k_paths):
        """Generate multiple chain-of-thought reasoning paths"""
        paths = []
        
        for i in range(k_paths):
            # Vary temperature for diversity
            temperature = 0.3 + (i * 0.2)
            
            prompt = f"""Question: {problem['question']}
Let's think step by step (attempt {i+1}):"""
            
            # In real implementation: response, logits = model.generate(prompt, return_logits=True)
            response = self.model.generate(prompt, temperature=temperature)
            
            # Mock logits for demonstration
            mock_logits = np.random.randn(10, 50257)
            
            paths.append({
                'answer': response,
                'logits': mock_logits,
                'temperature': temperature
            })
        
        return paths
    
    def _compute_path_entropies(self, paths):
        """Compute entropy for each reasoning path"""
        entropies = []
        
        for path in paths:
            # Compute entropy from logits
            logits = path['logits']
            
            # Average entropy across sequence
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
            path_entropy = np.mean([entropy(p) for p in probs])
            
            entropies.append(path_entropy)
        
        return entropies
    
    def _compute_entropy_confidence(self, probs):
        """Convert entropy to confidence score"""
        ent = entropy(probs)
        max_entropy = np.log(len(probs))
        
        # Low entropy = high confidence
        normalized_entropy = ent / max_entropy
        confidence = 1.0 - normalized_entropy
        
        return float(np.clip(confidence, 0.01, 0.99))
    
    def _reinforcement_selection(self, paths, entropies):
        """
        Select answer based on reinforcement learning
        Reinforce paths with low entropy (high confidence)
        """
        # Compute rewards (inverse of entropy)
        max_entropy = max(entropies) if entropies else 1.0
        rewards = [1.0 - (e / max_entropy) for e in entropies]
        
        # Store for learning
        self.reward_history.extend(rewards)
        
        # Select path with highest reward (lowest entropy)
        best_idx = np.argmax(rewards)
        selected_path = paths[best_idx]
        
        # Confidence based on reward and entropy
        confidence = rewards[best_idx]
        
        # Bonus for consistency across high-reward paths
        high_reward_paths = [i for i, r in enumerate(rewards) if r > self.entropy_threshold]
        if len(high_reward_paths) > 1:
            # Check answer consistency among high-reward paths
            answers = [paths[i]['answer'] for i in high_reward_paths]
            consistency = self._compute_answer_consistency(answers)
            confidence = confidence * (0.7 + 0.3 * consistency)
        
        return selected_path['answer'], float(np.clip(confidence, 0.01, 0.99))
    
    def _compute_answer_consistency(self, answers):
        """Compute consistency score among multiple answers"""
        if len(answers) <= 1:
            return 1.0
        
        # Simple consistency: check how many answers match the most common one
        from collections import Counter
        answer_counts = Counter(answers)
        most_common_count = answer_counts.most_common(1)[0][1]
        
        return most_common_count / len(answers)
    
    def _generate_reasoning_embeddings(self, paths, entropies):
        """Generate embeddings from selected reasoning paths"""
        embeddings = []
        
        # Select top paths based on entropy
        n_paths = min(5, len(paths))
        top_indices = np.argsort(entropies)[:n_paths]
        
        for idx in top_indices:
            path = paths[idx]
            reasoning_text = f"RENT path {idx}: {path['answer']}"
            embeddings.append(self.encoder.encode(reasoning_text))
        
        return embeddings

# ============================================================================
# CALIBRATION METRICS
# ============================================================================

class CalibrationMetrics:
    """Compute calibration metrics: ECE and Brier Score"""
    
    @staticmethod
    def expected_calibration_error(confidences, predictions, targets, n_bins=10):
        """Compute Expected Calibration Error (ECE)"""
        confidences = np.array(confidences)
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    @staticmethod
    def brier_score(confidences, targets):
        """Compute Brier Score"""
        confidences = np.array(confidences)
        targets = np.array(targets).astype(float)
        return np.mean((confidences - targets) ** 2)
    
    @staticmethod
    def selective_accuracy(confidences, predictions, targets, percentile=90):
        """Compute accuracy on high-confidence samples"""
        confidences = np.array(confidences)
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        threshold = np.percentile(confidences, percentile)
        high_conf_mask = confidences >= threshold
        
        if np.sum(high_conf_mask) == 0:
            return 0.0, 0.0
        
        selective_acc = (predictions[high_conf_mask] == targets[high_conf_mask]).mean()
        coverage = high_conf_mask.mean()
        
        return selective_acc, coverage

# ============================================================================
# DATASET LOADING
# ============================================================================

def load_aime_dataset(filepath='AIME2025.csv', limit=50):
    """Load AIME dataset"""
    try:
        df = pd.read_csv(filepath)
        questions = df['question'].dropna().astype(str).tolist()[:limit]
        
        answer_columns = ['answer', 'solution', 'correct_answer', 'result']
        answers = None
        
        for col in answer_columns:
            if col in df.columns:
                answers = df[col].dropna().astype(str).tolist()[:limit]
                break
        
        if answers is None:
            answers = [''] * len(questions)
        
        min_len = min(len(questions), len(answers))
        problems = []
        for i in range(min_len):
            problems.append({
                'question': questions[i],
                'answer': answers[i],
                'type': 'math'
            })
        
        print(f"Loaded {len(problems)} AIME problems")
        return problems
    except Exception as e:
        print(f"Error loading AIME: {e}, using synthetic data")
        return create_synthetic_math(limit)

def create_synthetic_math(n_samples):
    """Create synthetic math problems"""
    problems = []
    for i in range(n_samples):
        a, b = np.random.randint(10, 100, 2)
        problems.append({
            'question': f"What is {a} + {b}?",
            'answer': str(a + b),
            'type': 'math'
        })
    return problems

def convert_prediction_to_binary(prediction, problem):
    """Convert prediction to binary for evaluation"""
    pred_str = str(prediction).lower().strip()
    answer_str = str(problem['answer']).lower().strip()
    
    if problem['type'] == 'math':
        try:
            # Extract numbers from both prediction and answer
            pred_numbers = re.findall(r'\d+', pred_str)
            answer_numbers = re.findall(r'\d+', answer_str)
            
            if pred_numbers and answer_numbers:
                return 1 if pred_numbers[-1] == answer_numbers[0] else 0
        except:
            pass
    
    return 1 if pred_str == answer_str else 0

# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class ConfidenceBenchmark:
    """Main benchmark comparing confidence improvement methods"""
    
    def __init__(self):
        self.llama_model = LlamaModel("llama-8b")
        self.calibration_metrics = CalibrationMetrics()
        
        # Initialize all methods
        self.methods = {
            'GrACE': GrACEMethod(self.llama_model),
            'Credence': CredenceMethod(self.llama_model),
            'RENT': RENTMethod(self.llama_model),
            'Dirichlet+Topology (Ours)': self._create_topology_method()
        }
    
    def _create_topology_method(self):
        """Create wrapper for Dirichlet+Topology method"""
        class TopologyWrapper:
            def __init__(self, model):
                self.model = model
                self.name = "Dirichlet+Topology"
                self.topology_risk = EnhancedDirichletTopologyRisk()
                self.encoder = get_text_encoder()
            
            def solve_and_get_confidence(self, problem, dataset_type, k_reasoning_paths=7):
                # Generate answer
                prompt = f"Question: {problem['question']}\nAnswer:"
                answer = self.model.generate(prompt, temperature=0.3)
                
                # Generate diverse reasoning paths
                reasoning_embeddings = []
                for i in range(k_reasoning_paths):
                    temp = 0.4 + (i * 0.1)
                    reasoning_prompt = f"Analyze: {problem['question']}"
                    reasoning = self.model.generate(reasoning_prompt, temperature=temp)
                    reasoning_embeddings.append(self.encoder.encode(f"Path {i}: {reasoning}"))
                
                # Compute confidence using Dirichlet + Topology
                confidence_data = self.topology_risk.compute_confidence(
                    reasoning_embeddings, k_reasoning_paths
                )
                
                return answer, confidence_data['confidence'], reasoning_embeddings
        
        return TopologyWrapper(self.llama_model)
    
    def run_benchmark(self):
        """Run complete benchmark"""
        print("=" * 80)
        print("CONFIDENCE IMPROVEMENT METHODS BENCHMARK")
        print("=" * 80)
        print("Methods:")
        print("  1. GrACE - Generative confidence tokens")
        print("  2. Credence - Calibration game with iterative feedback")
        print("  3. RENT - RL with entropy-based reinforcement")
        print("  4. Dirichlet+Topology (Ours) - Risk-based confidence")
        print("=" * 80)
        
        # Load AIME dataset
        dataset = load_aime_dataset('AIME2025.csv', limit=30)
        
        results = {}
        
        # Test each method
        for method_name, method in self.methods.items():
            print(f"\nTesting {method_name}...")
            
            predictions = []
            targets = []
            confidences = []
            
            for i, problem in enumerate(dataset):
                try:
                    answer, confidence, reasoning = method.solve_and_get_confidence(
                        problem, 'text'
                    )
                    
                    pred_binary = convert_prediction_to_binary(answer, problem)
                    target_binary = 1  # Assume correct answer exists
                    
                    predictions.append(pred_binary)
                    targets.append(target_binary)
                    confidences.append(confidence)
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Processed {i+1}/{len(dataset)} samples")
                
                except Exception as e:
                    print(f"  Warning: Error on sample {i+1}: {e}")
                    continue
            
            # Compute metrics
            if predictions:
                metrics = self._compute_metrics(predictions, targets, confidences)
                results[method_name] = metrics
                self._print_results(method_name, metrics)
        
        # Generate comparison report
        self._generate_report(results)
        
        return results
    
    def _compute_metrics(self, predictions, targets, confidences):
        """Compute evaluation metrics"""
        predictions = np.array([int(p) for p in predictions])
        targets = np.array([int(t) for t in targets])
        confidences = np.array([float(c) for c in confidences])
        
        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average='binary', zero_division=0)
        mcc = matthews_corrcoef(targets, predictions)
        
        try:
            ece = self.calibration_metrics.expected_calibration_error(
                confidences, predictions, targets
            )
            brier = self.calibration_metrics.brier_score(confidences, targets)
        except:
            ece = 0.0
            brier = 0.5
        
        try:
            sel_acc_90, cov_90 = self.calibration_metrics.selective_accuracy(
                confidences, predictions, targets, 90
            )
            sel_acc_80, cov_80 = self.calibration_metrics.selective_accuracy(
                confidences, predictions, targets, 80
            )
        except:
            sel_acc_90, cov_90 = accuracy, 0.1
            sel_acc_80, cov_80 = accuracy, 0.2
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'mcc': mcc,
            'ece': ece,
            'brier': brier,
            'sel_acc_90': sel_acc_90,
            'coverage_90': cov_90,
            'sel_acc_80': sel_acc_80,
            'coverage_80': cov_80,
            'avg_confidence': np.mean(confidences),
            'n_samples': len(predictions)
        }
    
    def _print_results(self, method_name, metrics):
        """Print method results"""
        print(f"  {method_name}:")
        print(f"    Accuracy: {metrics['accuracy']:.3f}")
        print(f"    F1: {metrics['f1_score']:.3f}")
        print(f"    MCC: {metrics['mcc']:.3f}")
        print(f"    ECE: {metrics['ece']:.3f}")
        print(f"    Brier: {metrics['brier']:.3f}")
        print(f"    Selective Acc (90%): {metrics['sel_acc_90']:.3f}")
    
    def _generate_report(self, results):
        """Generate comparison report"""
        print(f"\n{'='*80}")
        print("COMPARISON REPORT")
        print(f"{'='*80}")
        
        print(f"\n{'Method':<35} {'Acc':<7} {'F1':<7} {'ECE':<7} {'Brier':<7}")
        print("-" * 80)
        
        for method, metrics in results.items():
            print(f"{method:<35} {metrics['accuracy']:.3f}   "
                  f"{metrics['f1_score']:.3f}   {metrics['ece']:.3f}   "
                  f"{metrics['brier']:.3f}")
        
        # Rank methods
        print(f"\n{'='*80}")
        print("METHOD RANKING (by composite score)")
        print(f"{'='*80}")
        
        scores = {}
        for method, m in results.items():
            # Composite: accuracy + (1-ECE) + (1-Brier)
            score = m['accuracy'] * 0.4 + (1 - m['ece']) * 0.3 + (1 - m['brier']) * 0.3
            scores[method] = score
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (method, score) in enumerate(ranked, 1):
            print(f"{i}. {method:<35} Score: {score:.3f}")
        
        print(f"\n{'='*80}")
        print("KEY INSIGHTS")
        print(f"{'='*80}")
        print("• GrACE: Model learns to generate calibrated confidence tokens")
        print("• Credence: Iterative feedback improves calibration dynamically")
        print("• RENT: RL reinforces high-confidence reasoning paths")
        print("• Dirichlet+Topology: Geometric risk assessment for confidence")
        print(f"\nBenchmark completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the benchmark"""
    print("Starting Confidence Improvement Methods Benchmark")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        benchmark = ConfidenceBenchmark()
        results = benchmark.run_benchmark()
        
        # Save results
        output_file = f"confidence_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json_results = {}
            for method, metrics in results.items():
                json_results[method] = {k: float(v) for k, v in metrics.items()}
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
