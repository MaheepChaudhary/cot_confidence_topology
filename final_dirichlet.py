import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

@dataclass
class DirichletConfig:
    embedding_dim: int = 768
    hidden_dim: int = 256
    dropout: float = 0.1
    reg_weight: float = 0.01
    min_alpha: float = 1.0
    max_alpha: float = 100.0
    num_cots: int = 3

class MultiHeadDirichletModel(nn.Module):
    """Multi-head model with separate 2D and 5D Dirichlet heads"""
    def __init__(self, cfg: DirichletConfig):
        super().__init__()
        self.cfg = cfg
        
        # Input for CoT features
        input_dim = cfg.embedding_dim * cfg.num_cots + cfg.num_cots + 2
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(), 
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim//2),
            nn.LayerNorm(cfg.hidden_dim//2),
            nn.ReLU(), 
            nn.Dropout(cfg.dropout),
        )
        
        # Separate heads for different dimensionalities
        self.binary_head = nn.Linear(cfg.hidden_dim//2, 2)    # 2D Dirichlet for binary questions
        self.multi_head = nn.Linear(cfg.hidden_dim//2, 5)     # 5D Dirichlet for 5-choice questions
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x, question_type):
        """
        Args:
            x: input features
            question_type: 'binary' or 'multi_choice'
        Returns:
            alpha: Dirichlet parameters (shape varies by question_type)
        """
        shared_features = self.shared_features(x)
        
        if question_type == 'binary':
            raw_alpha = self.binary_head(shared_features)
            alpha = torch.clamp(F.softplus(raw_alpha) + self.cfg.min_alpha, 
                              min=self.cfg.min_alpha, 
                              max=self.cfg.max_alpha)
            return alpha  # Shape: (batch, 2)
        
        elif question_type == 'multi_choice':
            raw_alpha = self.multi_head(shared_features)
            alpha = torch.clamp(F.softplus(raw_alpha) + self.cfg.min_alpha, 
                              min=self.cfg.min_alpha, 
                              max=self.cfg.max_alpha)
            return alpha  # Shape: (batch, 5)
        
        else:
            raise ValueError(f"Unknown question_type: {question_type}")

def dirichlet_loss_2d(alpha, labels, reg_weight=0.01):
    """Loss for 2D Dirichlet (binary questions)"""
    alpha = torch.clamp(alpha, min=1.0, max=100.0)
    
    alpha_sum = alpha.sum(dim=1, keepdim=True)
    probs = alpha / alpha_sum
    probs = torch.clamp(probs, min=1e-8, max=1-1e-8)
    
    nll = F.nll_loss(torch.log(probs), labels)
    
    precision = alpha_sum.squeeze()
    precision_penalty = torch.mean((precision - 4.0)**2)  # Target precision for binary
    
    loss = nll + reg_weight * precision_penalty
    return loss, nll.detach(), precision_penalty.detach()

def dirichlet_loss_5d(alpha, labels, reg_weight=0.01):
    """Loss for 5D Dirichlet (5-choice questions)"""
    alpha = torch.clamp(alpha, min=1.0, max=100.0)
    
    alpha_sum = alpha.sum(dim=1, keepdim=True)
    probs = alpha / alpha_sum
    probs = torch.clamp(probs, min=1e-8, max=1-1e-8)
    
    nll = F.nll_loss(torch.log(probs), labels)
    
    precision = alpha_sum.squeeze()
    precision_penalty = torch.mean((precision - 8.0)**2)  # Higher target for 5-choice
    
    loss = nll + reg_weight * precision_penalty
    return loss, nll.detach(), precision_penalty.detach()

class CoTFeatureExtractor:
    """Extract disagreement features from Chain-of-Thought embeddings"""
    
    @staticmethod
    def extract_cot_features(cot_embeddings: torch.Tensor) -> torch.Tensor:
        """Extract features with numerical stability"""
        batch_size, num_cots, embedding_dim = cot_embeddings.shape
        device = cot_embeddings.device
        
        # Normalize embeddings to prevent extreme values
        cot_embeddings = F.normalize(cot_embeddings, p=2, dim=2)
        
        # Stack all CoT embeddings
        stacked = cot_embeddings.reshape(batch_size, -1)
        
        # Compute variance across CoTs
        variance_per_dim = torch.var(cot_embeddings, dim=1, unbiased=False)
        mean_variance = variance_per_dim.mean(dim=1, keepdim=True)
        
        # Compute pairwise similarities
        similarities = []
        for i in range(num_cots):
            for j in range(i+1, num_cots):
                sim = F.cosine_similarity(
                    cot_embeddings[:, i, :], 
                    cot_embeddings[:, j, :], 
                    dim=1
                ).unsqueeze(1)
                similarities.append(sim)
        
        if similarities:
            sim_tensor = torch.cat(similarities, dim=1)
            sim_variance = torch.var(sim_tensor, dim=1, keepdim=True, unbiased=False)
        else:
            sim_variance = torch.zeros(batch_size, 1, device=device)
        
        # Per-CoT variance
        per_cot_variance = []
        for i in range(num_cots):
            cot_var = torch.var(cot_embeddings[:, i, :], dim=1, keepdim=True, unbiased=False)
            per_cot_variance.append(cot_var)
        per_cot_var_tensor = torch.cat(per_cot_variance, dim=1)
        
        # Combine features with epsilon for stability
        features = torch.cat([
            stacked,
            per_cot_var_tensor + 1e-8,
            mean_variance + 1e-8,
            sim_variance + 1e-8
        ], dim=1)
        
        return features

class MultiHeadCoTDataset(Dataset):
    """Dataset for multi-head model with true 2D/5D separation"""
    
    def __init__(self, json_file_path: str, embedding_dim: int = 768, use_mock_embeddings: bool = True):
        import json
        
        print(f"Loading CoT samples from {json_file_path}...")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} CoT samples")
        
        processed_data = []
        
        for sample in self.data:
            dataset_name = sample.get('dataset', 'unknown').lower()
            correct_answer = sample.get('correct_answer', 'A')
            choices = sample.get('choices', [])
            
            # Dataset-based question type detection
            if dataset_name in ['sarcasm', 'bias', 'hallucination', 'sarcasm_detection', 'bias_detection', 'hallucination_detection']:
                # Binary questions - TRUE 2D Dirichlet
                question_type = 'binary'
                num_choices = 2
                
                # Map to binary classes
                if correct_answer in ['A', '0', 'True', 'true', True, 0]:
                    label = 0
                elif correct_answer in ['B', '1', 'False', 'false', False, 1]:
                    label = 1
                else:
                    label = 0  # Default
                
            elif dataset_name in ['aime', 'sp500', 'sp500_prediction', 'aime_math']:
                # Multi-choice questions - TRUE 5D Dirichlet
                question_type = 'multi_choice'
                num_choices = 5
                
                # Map A->0, B->1, C->2, D->3, E->4
                label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
                label = label_map.get(correct_answer, 0)
                
            else:
                # Unknown dataset - infer from choices
                if len(choices) >= 5:
                    question_type = 'multi_choice'
                    num_choices = 5
                    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
                    label = label_map.get(correct_answer, 0)
                else:
                    question_type = 'binary'
                    num_choices = 2
                    label = 0 if correct_answer in ['A', 'True', '0'] else 1
            
            # Create mock embeddings
            cot_text = sample.get('cot_reasoning', '')
            
            if use_mock_embeddings:
                base_embedding = torch.randn(embedding_dim) * 0.1
                text_hash = hash(cot_text) % 1000
                
                # Different variance for different question types
                variance_scale = 0.08 if num_choices == 5 else 0.05
                
                variation1 = base_embedding + torch.randn(embedding_dim) * variance_scale + (text_hash / 1000.0)
                variation2 = base_embedding + torch.randn(embedding_dim) * variance_scale + (text_hash / 2000.0)
                variation3 = base_embedding + torch.randn(embedding_dim) * variance_scale + (text_hash / 3000.0)
                
                cot_embeddings = torch.stack([variation1, variation2, variation3])
            else:
                cot_embeddings = torch.randn(3, embedding_dim) * 0.1
            
            processed_sample = {
                'cot_embeddings': cot_embeddings,
                'label': label,
                'question_type': question_type,
                'num_choices': num_choices,
                'sample_id': sample.get('sample_id', f'sample_{len(processed_data)}'),
                'dataset': dataset_name,
                'question': sample.get('question', ''),
                'cot_reasoning': cot_text,
                'correct_answer': correct_answer,
                'choices': choices
            }
            
            processed_data.append(processed_sample)
        
        self.data = processed_data
        self.embedding_dim = embedding_dim
        self.num_cots = 3
        
        # Statistics
        print(f"Processed data: {len(self.data)} samples")
        
        type_counts = {}
        dataset_counts = {}
        
        for sample in self.data:
            qt = sample['question_type']
            ds = sample['dataset']
            
            type_counts[qt] = type_counts.get(qt, 0) + 1
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
        
        print(f"Question types: {type_counts}")
        print(f"Datasets: {dataset_counts}")
        
        # Show mappings by type
        print("Sample label mappings:")
        print("  Binary datasets (2D Dirichlet):")
        for dataset in ['sarcasm', 'bias', 'hallucination', 'sarcasm_detection', 'bias_detection', 'hallucination_detection']:
            samples = [s for s in self.data if s['dataset'] == dataset]
            if samples:
                sample = samples[0]
                print(f"    {dataset}: '{sample['correct_answer']}' -> {sample['label']}")
        
        print("  Multi-choice datasets (5D Dirichlet):")
        for dataset in ['aime', 'sp500', 'sp500_prediction', 'aime_math']:
            samples = [s for s in self.data if s['dataset'] == dataset]
            if samples:
                sample = samples[0]
                print(f"    {dataset}: '{sample['correct_answer']}' -> {sample['label']}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        return {
            'cot_embeddings': sample['cot_embeddings'],
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'question_type': sample['question_type'],
            'num_choices': sample['num_choices'],
            'sample_id': sample['sample_id'],
            'dataset': sample['dataset'],
            'question': sample['question'],
            'cot_reasoning': sample['cot_reasoning'],
            'correct_answer': sample['correct_answer']
        }

def create_multihead_dataloader(json_file_path: str, batch_size: int = 16, shuffle: bool = True):
    """Create DataLoader with separate batching by question type"""
    dataset = MultiHeadCoTDataset(json_file_path)
    
    # Separate samples by type
    binary_samples = [i for i, sample in enumerate(dataset) if dataset.data[i]['question_type'] == 'binary']
    multi_samples = [i for i, sample in enumerate(dataset) if dataset.data[i]['question_type'] == 'multi_choice']
    
    print(f"Split: {len(binary_samples)} binary, {len(multi_samples)} multi-choice samples")
    
    # Create separate samplers
    binary_sampler = torch.utils.data.SubsetRandomSampler(binary_samples) if shuffle else None
    multi_sampler = torch.utils.data.SubsetRandomSampler(multi_samples) if shuffle else None
    
    # Create separate DataLoaders
    binary_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=binary_sampler,
        shuffle=(shuffle and binary_sampler is None)
    )
    
    multi_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=multi_sampler,
        shuffle=(shuffle and multi_sampler is None)
    )
    
    return binary_loader, multi_loader, dataset.embedding_dim, dataset.num_cots

def train_multihead_model(json_file_path: str, num_epochs: int = 10, batch_size: int = 16):
    """Train multi-head model with true 2D/5D Dirichlet distributions"""
    print("Training Multi-Head Dirichlet Model")
    print("2D Dirichlet: sarcasm, bias, hallucination")
    print("5D Dirichlet: AIME, sp500")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data with type separation
    try:
        binary_loader, multi_loader, embedding_dim, num_cots = create_multihead_dataloader(
            json_file_path, batch_size=batch_size, shuffle=True
        )
        print(f"Data loaded successfully!")
        print(f"  Binary batches: {len(binary_loader)}")
        print(f"  Multi-choice batches: {len(multi_loader)}")
        print(f"  Embedding dim: {embedding_dim}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Create multi-head model
    cfg = DirichletConfig(
        embedding_dim=embedding_dim,
        num_cots=num_cots,
        hidden_dim=256,
        dropout=0.1,
        reg_weight=0.01
    )
    
    model = MultiHeadDirichletModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    total_params = sum(p.numel() for p in model.parameters())
    binary_params = sum(p.numel() for p in model.binary_head.parameters())
    multi_params = sum(p.numel() for p in model.multi_head.parameters())
    shared_params = total_params - binary_params - multi_params
    
    print(f"Model created:")
    print(f"  Total parameters: {total_params}")
    print(f"  Shared parameters: {shared_params}")
    print(f"  Binary head parameters: {binary_params}")
    print(f"  Multi-choice head parameters: {multi_params}")
    
    # Training loop
    model.train()
    epoch_stats = []
    
    for epoch in range(num_epochs):
        # Combined epoch statistics
        epoch_losses = []
        epoch_nlls = []
        epoch_regs = []
        
        # Separate tracking
        binary_correct = 0
        binary_total = 0
        multi_correct = 0
        multi_total = 0
        
        # Create combined iterator
        all_batches = []
        
        # Add binary batches with type marker
        for batch in binary_loader:
            all_batches.append(('binary', batch))
        
        # Add multi-choice batches with type marker
        for batch in multi_loader:
            all_batches.append(('multi_choice', batch))
        
        # Shuffle the combined batches
        if len(all_batches) > 0:
            np.random.shuffle(all_batches)
        
        pbar = tqdm(all_batches, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_type, batch in pbar:
            if batch_type == 'binary':
                # Filter to ensure all samples in batch are binary
                binary_indices = [i for i, qt in enumerate(batch['question_type']) if qt == 'binary']
                if not binary_indices:
                    continue
                
                # Extract binary samples - fix the tensor extraction
                cot_embeddings = batch['cot_embeddings'][binary_indices].to(device)
                labels = batch['label'][binary_indices].to(device)
                
                # Extract features
                features = CoTFeatureExtractor.extract_cot_features(cot_embeddings)
                
                # Forward pass through binary head
                alpha = model(features, 'binary')  # Shape: (batch, 2)
                loss, nll, reg = dirichlet_loss_2d(alpha, labels, cfg.reg_weight)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Statistics
                epoch_losses.append(loss.item())
                epoch_nlls.append(nll.item())
                epoch_regs.append(reg.item())
                
                # Binary accuracy
                alpha_sum = alpha.sum(dim=1, keepdim=True)
                probs = alpha / alpha_sum
                predictions = torch.argmax(probs, dim=1)
                batch_correct = (predictions == labels).sum().item()
                binary_correct += batch_correct
                binary_total += labels.size(0)
                
                pbar.set_postfix({
                    'Type': 'Binary',
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100*binary_correct/max(binary_total,1):.1f}%'
                })
            
            elif batch_type == 'multi_choice':
                # Filter to ensure all samples in batch are multi-choice
                multi_indices = [i for i, qt in enumerate(batch['question_type']) if qt == 'multi_choice']
                if not multi_indices:
                    continue
                
                # Extract multi-choice samples - fix the tensor extraction
                cot_embeddings = batch['cot_embeddings'][multi_indices].to(device)
                labels = batch['label'][multi_indices].to(device)
                
                # Extract features
                features = CoTFeatureExtractor.extract_cot_features(cot_embeddings)
                
                # Forward pass through multi-choice head
                alpha = model(features, 'multi_choice')  # Shape: (batch, 5)
                loss, nll, reg = dirichlet_loss_5d(alpha, labels, cfg.reg_weight)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Statistics
                epoch_losses.append(loss.item())
                epoch_nlls.append(nll.item())
                epoch_regs.append(reg.item())
                
                # Multi-choice accuracy
                alpha_sum = alpha.sum(dim=1, keepdim=True)
                probs = alpha / alpha_sum
                predictions = torch.argmax(probs, dim=1)
                batch_correct = (predictions == labels).sum().item()
                multi_correct += batch_correct
                multi_total += labels.size(0)
                
                pbar.set_postfix({
                    'Type': 'Multi',
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100*multi_correct/max(multi_total,1):.1f}%'
                })
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        avg_nll = np.mean(epoch_nlls) if epoch_nlls else 0
        avg_reg = np.mean(epoch_regs) if epoch_regs else 0
        
        total_correct = binary_correct + multi_correct
        total_samples = binary_total + multi_total
        overall_accuracy = 100 * total_correct / max(total_samples, 1)
        
        binary_accuracy = 100 * binary_correct / max(binary_total, 1)
        multi_accuracy = 100 * multi_correct / max(multi_total, 1)
        
        epoch_stats.append({
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'avg_nll': avg_nll,
            'avg_reg': avg_reg,
            'overall_accuracy': overall_accuracy,
            'binary_accuracy': binary_accuracy,
            'multi_accuracy': multi_accuracy,
            'binary_samples': binary_total,
            'multi_samples': multi_total
        })
        
        scheduler.step(avg_loss)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average NLL: {avg_nll:.4f}")
        print(f"  Average Reg: {avg_reg:.4f}")
        print(f"  Overall Accuracy: {overall_accuracy:.1f}%")
        print(f"  Binary Accuracy: {binary_accuracy:.1f}% ({binary_total} samples)")
        print(f"  Multi-Choice Accuracy: {multi_accuracy:.1f}% ({multi_total} samples)")
    
    print("\nTraining Complete!")
    print("=" * 60)
    
    return model, epoch_stats

def evaluate_multihead_model(model, json_file_path: str, batch_size: int = 16):
    """Evaluate multi-head model with true dimensional separation"""
    print("Evaluating Multi-Head Model")
    print("=" * 40)
    
    device = next(model.parameters()).device
    binary_loader, multi_loader, _, _ = create_multihead_dataloader(
        json_file_path, batch_size=batch_size, shuffle=False
    )
    
    model.eval()
    
    # Separate results tracking
    binary_results = {'predictions': [], 'labels': [], 'confidences': [], 'uncertainties': [], 'alphas': []}
    multi_results = {'predictions': [], 'labels': [], 'confidences': [], 'uncertainties': [], 'alphas': []}
    
    # Evaluate binary questions (2D Dirichlet)
    with torch.no_grad():
        for batch in tqdm(binary_loader, desc="Evaluating Binary"):
            # Filter binary samples
            binary_indices = [i for i, qt in enumerate(batch['question_type']) if qt == 'binary']
            if not binary_indices:
                continue
                
            cot_embeddings = batch['cot_embeddings'][binary_indices].to(device)
            labels = batch['label'][binary_indices].to(device)
            
            features = CoTFeatureExtractor.extract_cot_features(cot_embeddings)
            alpha = model(features, 'binary')  # True 2D Dirichlet
            
            # Compute metrics for 2D Dirichlet
            alpha_sum = alpha.sum(dim=1, keepdim=True)
            probs = alpha / alpha_sum
            predictions = torch.argmax(probs, dim=1)
            confidence_scores = torch.max(probs, dim=1)[0]
            
            # 2D-specific uncertainty
            precision = alpha_sum.squeeze()
            uncertainty = torch.clamp(1.0 - precision / 20.0, 0.0, 1.0)
            
            binary_results['predictions'].extend(predictions.cpu().numpy())
            binary_results['labels'].extend(labels.cpu().numpy())
            binary_results['confidences'].extend(confidence_scores.cpu().numpy())
            binary_results['uncertainties'].extend(uncertainty.cpu().numpy().flatten())
            binary_results['alphas'].extend(alpha.cpu().numpy())
        
        # Evaluate multi-choice questions (5D Dirichlet)
        for batch in tqdm(multi_loader, desc="Evaluating Multi-Choice"):
            # Filter multi-choice samples
            multi_indices = [i for i, qt in enumerate(batch['question_type']) if qt == 'multi_choice']
            if not multi_indices:
                continue
                
            cot_embeddings = batch['cot_embeddings'][multi_indices].to(device)
            labels = batch['label'][multi_indices].to(device)
            
            features = CoTFeatureExtractor.extract_cot_features(cot_embeddings)
            alpha = model(features, 'multi_choice')  # True 5D Dirichlet
            
            # Compute metrics for 5D Dirichlet
            alpha_sum = alpha.sum(dim=1, keepdim=True)
            probs = alpha / alpha_sum
            predictions = torch.argmax(probs, dim=1)
            confidence_scores = torch.max(probs, dim=1)[0]
            
            # 5D-specific uncertainty
            precision = alpha_sum.squeeze()
            uncertainty = torch.clamp(1.0 - precision / 50.0, 0.0, 1.0)
            
            multi_results['predictions'].extend(predictions.cpu().numpy())
            multi_results['labels'].extend(labels.cpu().numpy())
            multi_results['confidences'].extend(confidence_scores.cpu().numpy())
            multi_results['uncertainties'].extend(uncertainty.cpu().numpy().flatten())
            multi_results['alphas'].extend(alpha.cpu().numpy())
    
    # Calculate metrics
    print("\nEvaluation Results:")
    
    if binary_results['predictions']:
        binary_acc = np.mean(np.array(binary_results['predictions']) == np.array(binary_results['labels']))
        binary_conf = np.mean(binary_results['confidences'])
        binary_unc = np.mean(binary_results['uncertainties'])
        print(f"\n2D Dirichlet (Binary Questions):")
        print(f"  Datasets: sarcasm, bias, hallucination")
        print(f"  Accuracy: {binary_acc:.3f}")
        print(f"  Avg Confidence: {binary_conf:.3f}")
        print(f"  Avg Uncertainty: {binary_unc:.3f}")
        print(f"  Sample Count: {len(binary_results['predictions'])}")
        
        # Show alpha distribution for binary
        binary_alphas = np.array(binary_results['alphas'])
        print(f"  Alpha Stats: α1={binary_alphas[:,0].mean():.2f}±{binary_alphas[:,0].std():.2f}, "
              f"α2={binary_alphas[:,1].mean():.2f}±{binary_alphas[:,1].std():.2f}")
    
    if multi_results['predictions']:
        multi_acc = np.mean(np.array(multi_results['predictions']) == np.array(multi_results['labels']))
        multi_conf = np.mean(multi_results['confidences'])
        multi_unc = np.mean(multi_results['uncertainties'])
        print(f"\n5D Dirichlet (Multi-Choice Questions):")
        print(f"  Datasets: AIME, sp500")
        print(f"  Accuracy: {multi_acc:.3f}")
        print(f"  Avg Confidence: {multi_conf:.3f}")
        print(f"  Avg Uncertainty: {multi_unc:.3f}")
        print(f"  Sample Count: {len(multi_results['predictions'])}")
        
        # Show alpha distribution for multi-choice
        multi_alphas = np.array(multi_results['alphas'])
        alpha_means = [multi_alphas[:,i].mean() for i in range(5)]
        alpha_stds = [multi_alphas[:,i].std() for i in range(5)]
        alpha_str = ", ".join([f"α{i+1}={alpha_means[i]:.2f}±{alpha_stds[i]:.2f}" for i in range(5)])
        print(f"  Alpha Stats: {alpha_str}")
    
    return {'binary': binary_results, 'multi_choice': multi_results}

def save_distributions_to_json(model, json_file_path: str, output_file: str = "dirichlet_distributions.json", batch_size: int = 16):
    """Save all Dirichlet distributions and predictions to JSON file"""
    import json
    from datetime import datetime
    
    print(f"Saving distributions to {output_file}...")
    
    device = next(model.parameters()).device
    dataset = MultiHeadCoTDataset(json_file_path)
    model.eval()
    
    # Create comprehensive results structure
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'MultiHeadDirichletModel',
            'total_samples': len(dataset),
            'binary_samples': sum(1 for s in dataset.data if s['question_type'] == 'binary'),
            'multi_choice_samples': sum(1 for s in dataset.data if s['question_type'] == 'multi_choice'),
            'datasets': list(set(s['dataset'] for s in dataset.data)),
            'model_config': {
                'embedding_dim': dataset.embedding_dim,
                'num_cots': dataset.num_cots,
                'min_alpha': 1.0,
                'max_alpha': 100.0
            }
        },
        'samples': []
    }
    
    # Process all samples individually for complete information
    with torch.no_grad():
        for idx, sample_data in enumerate(tqdm(dataset, desc="Processing samples")):
            # Get original sample info
            original_sample = dataset.data[idx]
            
            cot_embeddings = sample_data['cot_embeddings'].unsqueeze(0).to(device)
            label = sample_data['label'].item()
            question_type = sample_data['question_type']
            
            # Extract features and get predictions
            features = CoTFeatureExtractor.extract_cot_features(cot_embeddings)
            alpha = model(features, question_type)
            
            # Compute all metrics
            alpha_sum = alpha.sum(dim=1, keepdim=True)
            probs = alpha / alpha_sum
            prediction = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs, dim=1)[0].item()
            precision = alpha_sum.squeeze().item()
            
            # Question type specific uncertainty
            if question_type == 'binary':
                uncertainty = max(0.0, min(1.0, 1.0 - precision / 20.0))
                target_precision = 4.0
            else:  # multi_choice
                uncertainty = max(0.0, min(1.0, 1.0 - precision / 50.0))
                target_precision = 8.0
            
            # Create sample result
            sample_result = {
                # Original sample information
                'sample_id': original_sample['sample_id'],
                'dataset': original_sample['dataset'],
                'question': original_sample['question'],
                'cot_reasoning': original_sample['cot_reasoning'],
                'choices': original_sample['choices'],
                'correct_answer': original_sample['correct_answer'],
                'true_label': label,
                
                # Model type and configuration
                'question_type': question_type,
                'num_choices': original_sample['num_choices'],
                'dirichlet_dimension': 2 if question_type == 'binary' else 5,
                
                # Dirichlet distribution parameters
                'alpha_parameters': alpha[0].cpu().numpy().tolist(),
                'alpha_sum': precision,
                'target_precision': target_precision,
                
                # Predictions and probabilities
                'probabilities': probs[0].cpu().numpy().tolist(),
                'predicted_label': prediction,
                'predicted_answer': _label_to_answer(prediction, question_type),
                
                # Confidence and uncertainty metrics
                'confidence': confidence,
                'uncertainty': uncertainty,
                'precision_score': precision,
                
                # Correctness
                'is_correct': prediction == label,
                'error_type': _get_error_type(prediction, label, question_type),
                
                # Additional analysis
                'distribution_concentration': _analyze_concentration(alpha[0].cpu().numpy()),
                'entropy': _calculate_entropy(probs[0].cpu().numpy())
            }
            
            results['samples'].append(sample_result)
    
    # Add summary statistics
    results['summary'] = _calculate_summary_stats(results['samples'])
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(results['samples'])} sample distributions to {output_file}")
    print(f"Binary samples: {results['metadata']['binary_samples']}")
    print(f"Multi-choice samples: {results['metadata']['multi_choice_samples']}")
    print(f"Overall accuracy: {results['summary']['overall_accuracy']:.3f}")
    
    return results

def _label_to_answer(label: int, question_type: str) -> str:
    """Convert numeric label back to answer format"""
    if question_type == 'binary':
        return 'A' if label == 0 else 'B'
    else:  # multi_choice
        return ['A', 'B', 'C', 'D', 'E'][label]

def _get_error_type(predicted: int, actual: int, question_type: str) -> str:
    """Classify the type of error"""
    if predicted == actual:
        return 'correct'
    elif question_type == 'binary':
        return 'binary_misclassification'
    else:
        return f'multi_choice_error_{abs(predicted - actual)}_steps'

def _analyze_concentration(alpha_array) -> dict:
    """Analyze how concentrated the Dirichlet distribution is"""
    alpha_sum = alpha_array.sum()
    alpha_max = alpha_array.max()
    alpha_min = alpha_array.min()
    
    return {
        'total_concentration': float(alpha_sum),
        'max_alpha': float(alpha_max),
        'min_alpha': float(alpha_min),
        'concentration_ratio': float(alpha_max / alpha_min) if alpha_min > 0 else float('inf'),
        'is_uniform': bool(abs(alpha_max - alpha_min) < 1.0),
        'dominant_class': int(alpha_array.argmax())
    }

def _calculate_entropy(probs) -> float:
    """Calculate Shannon entropy of probability distribution"""
    # Avoid log(0) by adding small epsilon
    probs_safe = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs_safe * np.log(probs_safe))
    return float(entropy)

def _calculate_summary_stats(samples) -> dict:
    """Calculate summary statistics across all samples"""
    binary_samples = [s for s in samples if s['question_type'] == 'binary']
    multi_samples = [s for s in samples if s['question_type'] == 'multi_choice']
    
    def calc_stats(sample_list, name):
        if not sample_list:
            return {}
        
        accuracies = [s['is_correct'] for s in sample_list]
        confidences = [s['confidence'] for s in sample_list]
        uncertainties = [s['uncertainty'] for s in sample_list]
        precisions = [s['precision_score'] for s in sample_list]
        entropies = [s['entropy'] for s in sample_list]
        
        return {
            f'{name}_count': len(sample_list),
            f'{name}_accuracy': sum(accuracies) / len(accuracies),
            f'{name}_avg_confidence': sum(confidences) / len(confidences),
            f'{name}_avg_uncertainty': sum(uncertainties) / len(uncertainties),
            f'{name}_avg_precision': sum(precisions) / len(precisions),
            f'{name}_avg_entropy': sum(entropies) / len(entropies),
            f'{name}_confidence_range': [min(confidences), max(confidences)],
            f'{name}_precision_range': [min(precisions), max(precisions)]
        }
    
    summary = {}
    summary.update(calc_stats(binary_samples, 'binary'))
    summary.update(calc_stats(multi_samples, 'multi_choice'))
    
    # Overall stats
    all_correct = [s['is_correct'] for s in samples]
    summary['overall_accuracy'] = sum(all_correct) / len(all_correct) if all_correct else 0
    summary['total_samples'] = len(samples)
    
    return summary

def demo_multihead_inference(model, json_file_path: str, num_samples: int = 6):
    """Demo inference showing true 2D vs 5D Dirichlet distributions"""
    print("\nDEMO: Multi-Head Inference (True 2D vs 5D Dirichlet)")
    print("=" * 60)
    
    device = next(model.parameters()).device
    dataset = MultiHeadCoTDataset(json_file_path)
    model.eval()
    
    # Get samples from each dataset type
    binary_samples = []
    multi_samples = []
    
    for i, sample in enumerate(dataset):
        if sample['question_type'] == 'binary' and len(binary_samples) < 3:
            binary_samples.append((i, sample))
        elif sample['question_type'] == 'multi_choice' and len(multi_samples) < 3:
            multi_samples.append((i, sample))
        
        if len(binary_samples) >= 3 and len(multi_samples) >= 3:
            break
    
    all_samples = binary_samples + multi_samples
    
    for idx, (sample_idx, sample) in enumerate(all_samples[:num_samples]):
        print(f"\nSample {idx+1}: {sample['sample_id']}")
        print(f"   Dataset: {sample['dataset']}")
        print(f"   Question Type: {sample['question_type']}")
        print(f"   Correct Answer: {sample['correct_answer']}")
        print(f"   True Label: {sample['label'].item()}")
        
        cot_embeddings = sample['cot_embeddings'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = CoTFeatureExtractor.extract_cot_features(cot_embeddings)
            
            question_type = sample['question_type']
            alpha = model(features, question_type)
            
            if question_type == 'binary':
                # True 2D Dirichlet inference
                alpha_sum = alpha.sum(dim=1, keepdim=True)
                probs = alpha / alpha_sum
                prediction = torch.argmax(probs, dim=1)
                confidence = torch.max(probs, dim=1)[0]
                precision = alpha_sum.squeeze()
                uncertainty = torch.clamp(1.0 - precision / 20.0, 0.0, 1.0)
                
                print(f"   TRUE 2D Dirichlet Alpha: {alpha[0].cpu().numpy().round(3)}")
                print(f"   Binary Probabilities: {probs[0].cpu().numpy().round(3)}")
                print(f"   Prediction: Class {prediction.item()} ({'True' if prediction.item()==0 else 'False'})")
                
            elif question_type == 'multi_choice':
                # True 5D Dirichlet inference
                alpha_sum = alpha.sum(dim=1, keepdim=True)
                probs = alpha / alpha_sum
                prediction = torch.argmax(probs, dim=1)
                confidence = torch.max(probs, dim=1)[0]
                precision = alpha_sum.squeeze()
                uncertainty = torch.clamp(1.0 - precision / 50.0, 0.0, 1.0)
                
                print(f"   TRUE 5D Dirichlet Alpha: {alpha[0].cpu().numpy().round(3)}")
                print(f"   All Probabilities: {probs[0].cpu().numpy().round(3)}")
                choice_labels = ['A', 'B', 'C', 'D', 'E']
                print(f"   Prediction: Class {prediction.item()} (Choice {choice_labels[prediction.item()]})")
            
            print(f"   Confidence: {confidence.item():.3f}")
            print(f"   Uncertainty: {uncertainty.item():.3f}")
            print(f"   Precision (Evidence): {precision.item():.3f}")
            
            # Correctness
            correct = "CORRECT" if prediction.item() == sample['label'].item() else "❌ INCORRECT"
            print(f"   Result: {correct}")

def plot_multihead_training(epoch_stats):
    """Plot training progress for multi-head model"""
    if not epoch_stats:
        print("No training statistics to plot")
        return
        
    epochs = [stat['epoch'] for stat in epoch_stats]
    losses = [stat['avg_loss'] for stat in epoch_stats]
    nlls = [stat['avg_nll'] for stat in epoch_stats]
    regs = [stat['avg_reg'] for stat in epoch_stats]
    overall_acc = [stat['overall_accuracy'] for stat in epoch_stats]
    binary_acc = [stat['binary_accuracy'] for stat in epoch_stats]
    multi_acc = [stat['multi_accuracy'] for stat in epoch_stats]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss components
    ax1.plot(epochs, losses, 'b-', label='Total Loss', linewidth=2)
    ax1.plot(epochs, nlls, 'r--', label='NLL', linewidth=2)
    ax1.plot(epochs, regs, 'g:', label='Regularization', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Multi-Head Training Loss Components')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy by type
    ax2.plot(epochs, overall_acc, 'purple', linewidth=3, label='Overall')
    ax2.plot(epochs, binary_acc, 'blue', linewidth=2, label='Binary (2D Dirichlet)')
    ax2.plot(epochs, multi_acc, 'red', linewidth=2, label='Multi-Choice (5D Dirichlet)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy by Question Type')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Sample distribution
    binary_samples = [stat['binary_samples'] for stat in epoch_stats]
    multi_samples = [stat['multi_samples'] for stat in epoch_stats]
    
    ax3.bar(['Binary\n(2D Dirichlet)', 'Multi-Choice\n(5D Dirichlet)'], 
           [binary_samples[-1], multi_samples[-1]], 
           color=['blue', 'red'], alpha=0.7)
    ax3.set_ylabel('Number of Samples')
    ax3.set_title('Sample Distribution by Type')
    ax3.grid(True, alpha=0.3)
    
    # Final performance comparison
    categories = ['Binary\n(sarcasm, bias,\nhallucination)', 'Multi-Choice\n(AIME, sp500)']
    final_accs = [binary_acc[-1], multi_acc[-1]]
    bars = ax4.bar(categories, final_accs, color=['blue', 'red'], alpha=0.7)
    ax4.set_ylabel('Final Accuracy (%)')
    ax4.set_title('Final Performance by Dataset Type')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, final_accs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function for multi-head Dirichlet training pipeline"""
    print("Multi-Head Dirichlet Training Pipeline")
    print("TRUE 2D Dirichlet: sarcasm, bias, hallucination")
    print("TRUE 5D Dirichlet: AIME, sp500")
    print("=" * 60)
    
    # Configuration
    JSON_FILE = "cot_samples.json"
    NUM_EPOCHS = 15
    BATCH_SIZE = 8
    
    try:
        print("Step 1: Training multi-head model...")
        model, training_stats = train_multihead_model(
            json_file_path=JSON_FILE,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE
        )
        
        print("\nStep 2: Plotting training progress...")
        plot_multihead_training(training_stats)
        
        print("\nStep 3: Evaluating by distribution type...")
        eval_results = evaluate_multihead_model(model, JSON_FILE, batch_size=BATCH_SIZE)
        
        print("\nStep 4: Saving all distributions to JSON...")
        distribution_data = save_distributions_to_json(model, JSON_FILE, "dirichlet_distributions.json", batch_size=BATCH_SIZE)
        
        print("\nStep 5: Demonstrating true dimensional inference...")
        demo_multihead_inference(model, JSON_FILE, num_samples=6)
        
        print("\n" + "=" * 60)
        print("MULTI-HEAD PIPELINE COMPLETE!")
        print("TRUE 2D Dirichlet for binary questions")
        print("TRUE 5D Dirichlet for multi-choice questions") 
        print("Separate parameter spaces for each type")
        print("No wasted dimensions or masking needed")
        print("Mathematically correct Dirichlet distributions")
        print("=" * 60)
        
        return model, training_stats, eval_results
        
    except FileNotFoundError:
        print(f"Error: Could not find {JSON_FILE}")
        print("Please make sure your JSON file is in the current directory")
        return None, None, None
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Run the multi-head pipeline
if __name__ == "__main__":
    model, stats, results = main()
