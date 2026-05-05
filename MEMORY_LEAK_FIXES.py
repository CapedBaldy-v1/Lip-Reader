"""
MEMORY LEAK FIXES FOR train_swin_vallr.py

Apply these fixes to prevent VRAM and RAM leaks that cause crashes after 5-10 epochs.

IDENTIFIED MEMORY LEAKS:
1. step_metrics list grows unbounded (stores every training step)
2. epoch_metrics list grows unbounded  
3. val_losses list grows unbounded
4. best_metrics_history list grows unbounded
5. sample_results list in validation stores all predictions
6. Tensors not properly detached before .item()
7. No garbage collection between epochs
8. TensorBoard writer accumulates data
9. Confusion matrix and phoneme stats accumulate
10. Figure generation keeps matplotlib objects in memory
"""

# ============================================================================
# FIX 1: Add memory cleanup after each epoch
# ============================================================================
# Add this import at the top of train_swin_vallr.py
import gc

# Add this method to the Trainer class:
def _cleanup_memory(self):
    """Force garbage collection and clear GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    LOGGER.debug("Memory cleanup: garbage collected and GPU cache cleared")

# ============================================================================
# FIX 2: Limit list sizes to prevent unbounded growth
# ============================================================================
# In Trainer.__init__, add these limits:
self.max_step_metrics = 10000  # Keep only last 10k steps
self.max_epoch_metrics = 1000  # Keep only last 1000 epochs
self.max_val_losses = 1000     # Keep only last 1000 validations

# ============================================================================
# FIX 3: Trim lists when they exceed limits
# ============================================================================
# Add this method to Trainer class:
def _trim_metrics_lists(self):
    """Trim metrics lists to prevent memory growth."""
    if len(self.step_metrics) > self.max_step_metrics:
        self.step_metrics = self.step_metrics[-self.max_step_metrics:]
        LOGGER.debug("Trimmed step_metrics to %d entries", len(self.step_metrics))
    
    if len(self.epoch_metrics) > self.max_epoch_metrics:
        self.epoch_metrics = self.epoch_metrics[-self.max_epoch_metrics:]
        LOGGER.debug("Trimmed epoch_metrics to %d entries", len(self.epoch_metrics))
    
    if len(self.val_losses) > self.max_val_losses:
        self.val_losses = self.val_losses[-self.max_val_losses:]
        LOGGER.debug("Trimmed val_losses to %d entries", len(self.val_losses))

# ============================================================================
# FIX 4: Properly detach tensors before storing
# ============================================================================
# In train_epoch(), change this line:
self.step_metrics.append(
    (
        self.global_step,
        loss.item(),  # Already detached
        ctc_loss.item(),  # Already detached
        decor_loss.item(),  # Already detached
        smooth_loss.item(),  # Already detached
        attn_loss.item()  # Already detached
    )
)

# To this (ensure tensors are detached):
self.step_metrics.append(
    (
        self.global_step,
        float(loss.detach().cpu().item()),
        float(ctc_loss.detach().cpu().item()),
        float(decor_loss.detach().cpu().item()),
        float(smooth_loss.detach().cpu().item()),
        float(attn_loss.detach().cpu().item())
    )
)

# ============================================================================
# FIX 5: Limit validation sample storage
# ============================================================================
# In validate(), change:
sample_results: List[Dict] = []

# To:
sample_results: List[Dict] = []
max_samples_to_store = 100  # Only store first 100 samples

# And change the append:
if len(sample_results) < max_samples_to_store:
    sample_results.append({
        "target":    target_text,
        "predicted": pred_str,
        "cer":       round(cer, 4),
        "wer":       round(wer, 4),
    })

# ============================================================================
# FIX 6: Clear matplotlib figures properly
# ============================================================================
# In _generate_best_figures(), after each plt.close(fig), add:
del fig, ax
gc.collect()

# ============================================================================
# FIX 7: Flush TensorBoard writer periodically
# ============================================================================
# Add this method to Trainer class:
def _flush_tensorboard(self):
    """Flush TensorBoard writer to disk."""
    try:
        self.writer.flush()
        LOGGER.debug("TensorBoard writer flushed")
    except Exception:
        log_exception(LOGGER, "Failed to flush TensorBoard writer")

# ============================================================================
# FIX 8: Clear validation results after figure generation
# ============================================================================
# In _generate_best_figures(), at the end add:
# Clear validation results to free memory
self._last_val_results = None
gc.collect()

# ============================================================================
# FIX 9: Delete intermediate tensors explicitly
# ============================================================================
# In train_epoch(), after backward pass, add:
# Explicitly delete tensors to free memory
del video, logits, features, loss, ctc_loss, decor_loss, smooth_loss, attn_loss
if 'attn_weights' in locals():
    del attn_weights

# ============================================================================
# FIX 10: Call cleanup at end of each epoch
# ============================================================================
# At the end of train_epoch(), before return, add:
self._trim_metrics_lists()
self._cleanup_memory()

# At the end of validate(), before return, add:
self._cleanup_memory()
