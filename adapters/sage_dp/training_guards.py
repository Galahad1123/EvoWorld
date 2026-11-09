"""Keep validation from advancing the original fork's LR scheduler."""
def run_epoch_guarded(original,*args,**kwargs):
    model=kwargs['model'];callback=model.on_gradient_step
    if kwargs.get('validate',False):model.on_gradient_step=lambda:None
    try:return original(*args,**kwargs)
    finally:model.on_gradient_step=callback
