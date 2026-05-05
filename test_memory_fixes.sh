#!/bin/bash
#
# Test memory leak fixes
#

echo "Testing memory leak fixes..."
echo ""

# Check if fixes are applied
echo "Checking if fixes are applied to train_swin_vallr.py:"
echo ""

echo -n "  ✓ gc import: "
grep -q "^import gc" train_swin_vallr.py && echo "FOUND" || echo "MISSING"

echo -n "  ✓ _cleanup_memory method: "
grep -q "def _cleanup_memory" train_swin_vallr.py && echo "FOUND" || echo "MISSING"

echo -n "  ✓ _trim_metrics_lists method: "
grep -q "def _trim_metrics_lists" train_swin_vallr.py && echo "FOUND" || echo "MISSING"

echo -n "  ✓ max_step_metrics limit: "
grep -q "self.max_step_metrics = 10000" train_swin_vallr.py && echo "FOUND" || echo "MISSING"

echo -n "  ✓ Tensor detachment in step_metrics: "
grep -q "float(loss.detach().cpu().item())" train_swin_vallr.py && echo "FOUND" || echo "MISSING"

echo -n "  ✓ Explicit tensor deletion: "
grep -q "del video, logits, features" train_swin_vallr.py && echo "FOUND" || echo "MISSING"

echo -n "  ✓ max_samples_to_store limit: "
grep -q "max_samples_to_store = 100" train_swin_vallr.py && echo "FOUND" || echo "MISSING"

echo -n "  ✓ Cleanup at end of train_epoch: "
grep -q "self._trim_metrics_lists()" train_swin_vallr.py && echo "FOUND" || echo "MISSING"

echo -n "  ✓ Cleanup at end of validate: "
grep -A5 "return avg_loss, avg_cer" train_swin_vallr.py | grep -q "_cleanup_memory" && echo "FOUND" || echo "MISSING"

echo -n "  ✓ Clear validation results after figures: "
grep -q "self._last_val_results = None" train_swin_vallr.py && echo "FOUND" || echo "MISSING"

echo ""
echo "All fixes checked!"
echo ""
echo "To monitor memory during training:"
echo "  Terminal 1: bash FINAL_TRAINING_COMMAND.sh"
echo "  Terminal 2: watch -n 1 'free -h'"
echo "  Terminal 3: watch -n 1 'rocm-smi | grep -A 5 Memory'"
echo ""
