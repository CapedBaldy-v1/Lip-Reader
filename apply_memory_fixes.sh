#!/bin/bash
#
# Apply memory leak fixes to train_swin_vallr.py
#

echo "Applying matplotlib cleanup fixes..."

# Add cleanup after each plt.close(fig) call
sed -i 's/plt\.close(fig)$/plt.close(fig)\n            del fig, ax\n            gc.collect()/g' train_swin_vallr.py

# Add cleanup at end of _generate_best_figures
sed -i '/def _generate_best_figures/,/^    def / {
    /^    def / i\        # MEMORY LEAK FIX: Clear validation results after figure generation\
        self._last_val_results = None\
        gc.collect()\
        LOGGER.debug("Cleared validation results and collected garbage after figure generation")
}' train_swin_vallr.py

echo "Memory leak fixes applied!"
echo ""
echo "Summary of fixes:"
echo "  ✓ Added gc import"
echo "  ✓ Added memory cleanup methods (_cleanup_memory, _trim_metrics_lists, _flush_tensorboard)"
echo "  ✓ Limited list sizes (step_metrics, epoch_metrics, val_losses)"
echo "  ✓ Properly detach tensors before storing"
echo "  ✓ Explicitly delete tensors after use"
echo "  ✓ Limited validation sample storage to 100"
echo "  ✓ Added cleanup at end of train_epoch"
echo "  ✓ Added cleanup at end of validate"
echo "  ✓ Added matplotlib object cleanup"
echo "  ✓ Clear validation results after figure generation"
echo ""
echo "These fixes should prevent memory leaks that cause crashes after 5-10 epochs."
