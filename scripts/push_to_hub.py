"""Upload a release model folder and its card to the Hugging Face Hub.

Dry run by default: prints the repo id, the private flag and the exact file list, and does not touch the network.
    python scripts/push_to_hub.py experiments/hf/4tokens kcsheang/ts-t5-base-4tokens \
        --card model_cards/ts-t5-base-4tokens.md [--public] [--no-dry-run]
A real upload needs a logged-in Hub session (`huggingface-cli login` or the HF_TOKEN variable); this script never reads a token.
"""
import argparse
from pathlib import Path

MODEL_FILES = ('model.safetensors', 'config.json', 'generation_config.json')
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'spiece.model', 'special_tokens_map.json')


def release_files(model_dir: Path, card: Path) -> list[tuple[Path, str]]:
    """(local path, path in repo) for every file to upload. The card goes up as README.md."""
    model_dir = Path(model_dir)
    for required in ('model.safetensors', 'config.json'):
        if not (model_dir / required).exists():
            raise FileNotFoundError(f'{model_dir / required} is missing')
    if not Path(card).exists():
        raise FileNotFoundError(f'card {card} is missing')
    files = [(model_dir / n, n) for n in MODEL_FILES + TOKENIZER_FILES if (model_dir / n).exists()]
    return files + [(Path(card), 'README.md')]


def push(model_dir: Path, repo_id: str, card: Path, private: bool = True, dry_run: bool = True) -> None:
    files = release_files(model_dir, card)
    print(f'repo: {repo_id}')
    print(f'private: {private}')
    print('files:')
    for local, remote in files:
        print(f'  {remote}  <-  {local}')
    if dry_run:
        print(f'would set visibility: {"private" if private else "public"} (also on an existing repo)')
        print('dry run: nothing uploaded (use --no-dry-run to upload)')
        return
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.update_repo_settings(repo_id, private=private)  # create_repo(exist_ok=True) leaves an existing repo's visibility alone
    api.create_commit(repo_id, operations=[CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(l)) for l, r in files],
                      commit_message='Add TS-T5 model, tokenizer and card')
    print(f'uploaded to {repo_id}')


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model_dir', type=Path)
    p.add_argument('repo_id')
    p.add_argument('--card', type=Path, required=True)
    p.add_argument('--public', action='store_true', help='make the repo public (default: private)')
    p.add_argument('--no-dry-run', action='store_true', help='really upload')
    a = p.parse_args(argv)
    push(a.model_dir, a.repo_id, a.card, private=not a.public, dry_run=not a.no_dry_run)


if __name__ == '__main__':
    main()
