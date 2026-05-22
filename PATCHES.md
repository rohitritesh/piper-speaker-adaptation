# Patches applied to Piper

This project bundles a copy of [Piper TTS](https://github.com/rhasspy/piper) under `piper/` so that the entire fine-tuning workflow stays self-contained and reproducible. Two small but important patches have been applied to that bundled copy. This file explains what was changed and why, so that anyone reading the code can verify the modifications are minimal and well-motivated.

The original (unmodified) version of the most important file is preserved alongside the patched version as `piper/src/python/piper_train/__main__.py.original`, in case anyone wants to inspect the original behaviour or roll back the patch.

## Why we needed to patch Piper

Piper's training code was written for **PyTorch Lightning 1.x**. The most recent Piper release on GitHub still depends on Lightning 1.7. We wanted to run on **PyTorch 2.6 + Lightning 2.4**, because:

- PyTorch 2.6 is the only version with a stable CUDA 12.4 build that supports recent NVIDIA drivers on Ada Lovelace laptops.
- Installing the old Lightning 1.x release alongside modern PyTorch causes import-time conflicts that are hard to resolve without downgrading the entire stack.
- Lightning 2.x removed two APIs that Piper's training code relied on, so the original code does not run on a modern stack without changes.

Patching Piper rather than downgrading the stack was the cleaner choice. The patches are small (a few dozen lines total) and the rest of Piper works unchanged.

## Patch 1: `piper_train/__main__.py`

**What changed:** the original file used two Lightning 1.x convenience methods that were removed in Lightning 2.x:

- `Trainer.add_argparse_args(parser)` (registered Lightning's CLI flags onto an `argparse.ArgumentParser`)
- `Trainer.from_argparse_args(args)` (built a `Trainer` instance from the parsed arguments)

**Replacement:** we declare the small set of Trainer arguments we actually need explicitly (`--accelerator`, `--devices`, `--max_epochs`, `--precision`, `--default_root_dir`, `--resume_from_checkpoint`, and a handful of others) and build the `Trainer` directly with `Trainer(**kwargs)`. We also moved the `resume_from_checkpoint` argument from the Trainer constructor into `trainer.fit(model, ckpt_path=...)`, which is where Lightning 2.x expects it.

**Behavioural equivalence:** the patched script accepts the same command-line flags Piper's original `__main__.py` did, and constructs an equivalent Trainer. The `ModelCheckpoint` callback is added in the same way the original code did, with the same `every_n_epochs` setting.

The original file is kept at `piper/src/python/piper_train/__main__.py.original` so anyone can diff the two.

## Patch 2: `piper_train/vits/lightning.py`

**What changed:** Lightning 2.x removed support for the "two optimizers with automatic alternation" pattern that Piper's VITS training relied on. In Lightning 1.x, `training_step` could declare an `optimizer_idx` parameter and Lightning would call it twice per step (once for the generator, once for the discriminator), alternating the optimizers automatically.

In Lightning 2.x, training loops with multiple optimizers must be written in **manual optimization mode**.

**Two small edits were made:**

1. In `VitsModel.__init__`, we added `self.automatic_optimization = False`. This tells Lightning that we will drive the optimizer steps ourselves.

2. We rewrote `training_step` to manually orchestrate the generator and discriminator updates:

   ```python
   def training_step(self, batch, batch_idx):
       opt_g, opt_d = self.optimizers()

       opt_g.zero_grad()
       loss_g = self.training_step_g(batch)
       self.manual_backward(loss_g)
       opt_g.step()

       opt_d.zero_grad()
       loss_d = self.training_step_d(batch)
       self.manual_backward(loss_d)
       opt_d.step()

       if self.trainer.is_last_batch:
           schedulers = self.lr_schedulers()
           if schedulers is not None:
               if not isinstance(schedulers, (list, tuple)):
                   schedulers = [schedulers]
               for sch in schedulers:
                   sch.step()

       total = loss_g.detach() + loss_d.detach()
       self.log('train_loss', total, prog_bar=True)
       return total
   ```

   The `training_step_g` and `training_step_d` helper methods that compute the actual losses were not touched. The patch only changes the wrapper that calls them, so the VITS training mathematics remain identical to the original Piper implementation.

**Why this preserves training behaviour:** in Lightning 1.x with automatic alternation, the framework was calling exactly this same sequence under the hood. Making the orchestration explicit means our code does the same work; we just write the order ourselves.

## How to verify the patches

To see the difference against an unmodified Piper checkout:

```bash
git clone https://github.com/rhasspy/piper.git /tmp/piper-upstream
diff piper/src/python/piper_train/__main__.py /tmp/piper-upstream/src/python/piper_train/__main__.py
diff piper/src/python/piper_train/vits/lightning.py /tmp/piper-upstream/src/python/piper_train/vits/lightning.py
```

The first diff will be substantial (we rewrote the argument parsing block). The second diff is small (two short additions, no deletions of training logic).

## Could we have done this differently?

Yes. A pure-downgrade path also works: install `pytorch-lightning==1.9.5` and Piper runs without any patches. We chose to patch instead because it keeps the rest of the project on the modern PyTorch 2.6 + Lightning 2.4 stack, which has noticeably better diagnostics, better CUDA support, and active maintenance. The patches are small, well-isolated, and the original file is preserved alongside as `__main__.py.original` for reference.
