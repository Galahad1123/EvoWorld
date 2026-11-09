"""Keep the SAC entropy scalar trainable at native small learning rates."""
def prepare_entropy_precision(worker):
 # Module.float preserves the existing optimizer's Parameter reference.
 worker.entropy_temp.float()
