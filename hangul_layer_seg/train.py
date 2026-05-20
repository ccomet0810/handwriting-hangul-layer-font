from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def import_training_deps():
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader

    try:
        from .dataset import HangulLayerDataset, train_val_split
        from .losses import MultiLabelLayerLoss
        from .models import UNet
        from .utils.layer_mask_io import LAYERS, make_layer_overlay
    except ImportError:
        from dataset import HangulLayerDataset, train_val_split
        from losses import MultiLabelLayerLoss
        from models import UNet
        from utils.layer_mask_io import LAYERS, make_layer_overlay
    return torch, Image, DataLoader, HangulLayerDataset, train_val_split, MultiLabelLayerLoss, UNet, LAYERS, make_layer_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3-channel sigmoid Hangul layer segmentation model.")
    parser.add_argument("--glyph_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--save_dir", default="runs/layer_unet_baseline")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dice_loss_weight", type=float, default=0.5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--cache_in_memory", action="store_true", help="Preload resized images and masks into RAM.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision when available.")
    parser.add_argument("--channels_last", action="store_true", help="Use channels_last memory format on CUDA.")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile for the model.")
    parser.add_argument("--val_interval", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument("--sample_interval", type=int, default=1, help="Write sample_predictions every N epochs. 0 disables.")
    parser.add_argument("--save_interval", type=int, default=1, help="Write checkpoint_latest every N epochs.")
    return parser.parse_args()


def resolve_device(torch, name: str):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(torch, device, enabled: bool):
    if not enabled:
        return torch.amp.autocast(device_type=device.type, enabled=False)
    return torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda")


def run_epoch(model, loader, criterion, device, torch, optimizer=None, scaler=None, amp: bool = False, channels_last: bool = False) -> float:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.set_grad_enabled(train):
            with autocast_context(torch, device, amp):
                logits = model(images)
                loss = criterion(logits, target, valid)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_count += images.size(0)
    return total_loss / max(total_count, 1)


def save_sample_grid(model, loader, device, out_path: Path, image_size: int, torch, Image, make_layer_overlay, amp: bool = False) -> None:
    model.eval()
    try:
        batch = next(iter(loader))
    except StopIteration:
        return
    with torch.no_grad(), autocast_context(torch, device, amp):
        logits = model(batch["image"].to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    imgs = (batch["image"].squeeze(1).numpy() * 255).clip(0, 255).astype("uint8")
    targets = batch["target"].numpy()

    cell_w = image_size
    labels_h = 22
    cols = 3
    rows = min(6, len(imgs))
    sheet = Image.new("RGB", (cols * cell_w, rows * (cell_w + labels_h)), (245, 245, 245))
    for row in range(rows):
        glyph = Image.fromarray(imgs[row], mode="L")
        gt = make_layer_overlay(glyph, targets[row] > 0.5)
        pred = make_layer_overlay(glyph, probs[row] > 0.5)
        for col, img in enumerate([glyph.convert("RGB"), gt, pred]):
            sheet.paste(img.resize((cell_w, cell_w), Image.NEAREST), (col * cell_w, row * (cell_w + labels_h) + labels_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def save_checkpoint(path: Path, model, optimizer, epoch: int, val_loss: float, args: argparse.Namespace, torch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_kwargs": {"in_channels": 1, "num_classes": 3, "base_channels": args.base_channels},
            "image_size": args.image_size,
            "layers": ["L", "V", "T"],
            "activation": "sigmoid",
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    (
        torch,
        Image,
        DataLoader,
        HangulLayerDataset,
        train_val_split,
        MultiLabelLayerLoss,
        UNet,
        _LAYERS,
        make_layer_overlay,
    ) = import_training_deps()
    torch.manual_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(torch, args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset = HangulLayerDataset(
        args.glyph_dir,
        args.mask_dir,
        image_size=args.image_size,
        cache_in_memory=args.cache_in_memory,
    )
    if len(dataset) == 0:
        raise SystemExit(f"No paired layer samples found under {args.glyph_dir} and {args.mask_dir}.")

    train_set, val_set = train_val_split(dataset, args.val_ratio, args.seed)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    model = UNet(in_channels=1, num_classes=3, base_channels=args.base_channels).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"torch.compile disabled: {type(exc).__name__}: {exc}")
    criterion = MultiLabelLayerLoss(dice_loss_weight=args.dice_loss_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_val = float("inf")
    last_val = float("nan")
    with (save_dir / "loss_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "seconds"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            train_loss = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                torch,
                optimizer,
                scaler=scaler,
                amp=args.amp,
                channels_last=args.channels_last and device.type == "cuda",
            )
            should_validate = epoch == 1 or epoch == args.epochs or (args.val_interval > 0 and epoch % args.val_interval == 0)
            if should_validate:
                last_val = run_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    torch,
                    amp=args.amp,
                    channels_last=args.channels_last and device.type == "cuda",
                )
            seconds = time.perf_counter() - start
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": last_val, "seconds": round(seconds, 3)})
            handle.flush()
            if args.save_interval > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.save_interval == 0):
                save_checkpoint(save_dir / "checkpoint_latest.pt", model, optimizer, epoch, last_val, args, torch)
            if should_validate and last_val < best_val:
                best_val = last_val
                save_checkpoint(save_dir / "checkpoint_best.pt", model, optimizer, epoch, last_val, args, torch)
            if args.sample_interval > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.sample_interval == 0):
                save_sample_grid(model, val_loader, device, save_dir / "sample_predictions.png", args.image_size, torch, Image, make_layer_overlay, amp=args.amp)
            print(f"epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.4f} val_loss={last_val:.4f} seconds={seconds:.2f}")


if __name__ == "__main__":
    main()
