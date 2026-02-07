"""DuneDog CLI — stochastic story generation from the command line."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from pydantic import SecretStr

from dunedog.models.config import GenerationConfig, Preset


console = Console()


def _bounded_int(lo: int, hi: int):
    """Return an argparse type function that enforces lo <= value <= hi."""
    def _parse(value: str) -> int:
        iv = int(value)
        if iv < lo or iv > hi:
            raise argparse.ArgumentTypeError(f"must be between {lo} and {hi}, got {value}")
        return iv
    _parse.__name__ = f"int[{lo}..{hi}]"
    return _parse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dunedog",
        description="DuneDog — stochastic story generation engine",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # -- generate --
    gen = sub.add_parser("generate", help="Full pipeline: chaos → skeletons → stories")
    gen.add_argument("-n", "--count", type=_bounded_int(1, 10000), default=None, help="Number of skeletons (1-10000, default from preset)")
    gen.add_argument("-o", "--output", type=str, default=None, help="Output file path")
    gen.add_argument("--preset", type=str, default="deep", choices=["quick", "deep", "experimental", "custom"])
    gen.add_argument("--provider", type=str, default=None, choices=["openai", "anthropic", "openrouter", "chatgpt"])
    gen.add_argument("--model", type=str, default=None, help="Model name (provider-specific)")
    gen.add_argument("--api-key", type=str, default=None, help="API key (prefer env var; visible in process list)")
    gen.add_argument("--story-lines", type=_bounded_int(1, 200), default=20, help="Max lines per story (1-200, default 20)")
    gen.add_argument("--stories-for-llm", type=_bounded_int(1, 20), default=20, help="Max skeletons sent to LLM (1-20, default 20)")
    gen.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    gen.add_argument("--no-llm", action="store_true", help="Skip LLM synthesis, export skeletons only")

    # -- demo --
    demo = sub.add_parser("demo", help="Generate and display a single skeleton")
    demo.add_argument("--seed", type=int, default=None, help="Random seed")
    demo.add_argument("--no-llm", action="store_true", help="Skip LLM synthesis")
    demo.add_argument("--provider", type=str, default=None)
    demo.add_argument("--model", type=str, default=None)
    demo.add_argument("--api-key", type=str, default=None)

    # -- soup --
    soup = sub.add_parser("soup", help="Generate letter soup only")
    soup.add_argument("-l", "--length", type=_bounded_int(1, 10000), default=200, help="Soup length (1-10000, default 200)")
    soup.add_argument("-n", "--count", type=_bounded_int(1, 10000), default=1, help="Number of soups (1-10000, default 1)")
    soup.add_argument("--seed", type=int, default=None)

    # -- evolve --
    evolve = sub.add_parser("evolve", help="Evolve existing skeletons")
    evolve.add_argument("-i", "--input", type=str, required=True, help="Input skeleton JSON")
    evolve.add_argument("-o", "--output", type=str, required=True, help="Output file")
    evolve.add_argument("-g", "--generations", type=_bounded_int(1, 1000), default=5, help="Number of generations (1-1000, default 5)")
    evolve.add_argument("--seed", type=int, default=None)

    # -- setup --
    sub.add_parser("setup", help="Download NLTK data and verify setup")

    return parser


def _resolve_api_key(provider: str | None, api_key: str | None) -> str:
    """Resolve API key from argument or environment variable."""
    if api_key:
        return api_key
    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "chatgpt": "CHATGPT_TOKEN",
    }
    if provider and provider in env_map:
        return os.environ.get(env_map[provider], "")
    return ""


def cmd_generate(args: argparse.Namespace) -> None:
    """Full generation pipeline."""
    from dunedog.utils.seed_manager import SeedManager
    from dunedog.output.batch_generator import StoryBatchGenerator
    from dunedog.output import exporter

    config = GenerationConfig.from_preset(args.preset, seed=args.seed)
    if args.count is not None:
        config.skeletons_to_generate = args.count
    config.llm.story_lines = args.story_lines
    config.llm.max_stories_for_llm = args.stories_for_llm

    if args.provider:
        config.llm.provider = args.provider
    if args.model:
        config.llm.model = args.model
    config.llm.api_key = SecretStr(_resolve_api_key(args.provider or config.llm.provider, args.api_key))

    seed_mgr = SeedManager(config.seed)
    generator = StoryBatchGenerator(config, seed_mgr)

    console.print(f"[bold]Generating {config.skeletons_to_generate} skeletons (preset: {args.preset})...[/bold]")
    skeletons = generator.generate_batch()

    if not skeletons:
        console.print("[red]No skeletons generated.[/red]")
        return

    console.print(f"[green]Generated {len(skeletons)} skeletons.[/green]")
    console.print(f"Best coherence score: {skeletons[0].coherence_score:.4f}")

    if args.no_llm:
        output_path = args.output or "skeletons.json"
        exporter.to_json(skeletons, output_path)
        console.print(f"Skeletons exported to [bold]{output_path}[/bold]")
        return

    # LLM synthesis
    if not config.llm.api_key.get_secret_value():
        console.print("[yellow]No API key provided. Use --api-key or set environment variable.[/yellow]")
        console.print("[yellow]Falling back to skeleton export.[/yellow]")
        output_path = args.output or "skeletons.json"
        exporter.to_json(skeletons, output_path)
        console.print(f"Skeletons exported to [bold]{output_path}[/bold]")
        return

    from dunedog.llm.provider import create_provider
    from dunedog.llm.synthesizer import StorySynthesizer

    provider = create_provider(
        config.llm.provider,
        api_key=config.llm.api_key.get_secret_value(),
        model=config.llm.model,
    )
    synthesizer = StorySynthesizer(provider, config.llm)

    console.print(f"[bold]Synthesizing stories via {config.llm.provider}...[/bold]")
    top_skeletons = skeletons[:config.llm.max_stories_for_llm]

    stories = asyncio.run(synthesizer.synthesize(top_skeletons))

    if stories:
        console.print(f"\n[bold green]Generated {len(stories)} stories:[/bold green]\n")
        for story in stories:
            console.print(Panel(
                Text(story.content),
                title=Text(story.title, style="bold"),
                subtitle=f"Strategy: {story.strategy}",
                border_style="blue",
            ))
            console.print()

        output_path = args.output or "stories.json"
        exporter.export_stories([s.to_dict() for s in stories], output_path)
        console.print(f"Stories exported to [bold]{output_path}[/bold]")
    else:
        console.print("[yellow]No stories returned from LLM.[/yellow]")
        output_path = args.output or "skeletons.json"
        exporter.to_json(skeletons, output_path)
        console.print(f"Skeletons exported to [bold]{output_path}[/bold]")


def cmd_demo(args: argparse.Namespace) -> None:
    """Demo: generate and display a single skeleton."""
    from dunedog.utils.seed_manager import SeedManager
    from dunedog.output.batch_generator import StoryBatchGenerator

    config = GenerationConfig.from_preset("quick", seed=args.seed)
    config.skeletons_to_generate = 1

    seed_mgr = SeedManager(config.seed)
    generator = StoryBatchGenerator(config, seed_mgr)

    console.print("[bold]Generating demo skeleton...[/bold]\n")
    rng = seed_mgr.child_rng("demo")
    skeleton = generator.generate_single(rng)

    # Display primordial source
    ps = skeleton.primordial_source
    if ps.letter_soup_raw:
        soup_display = ps.letter_soup_raw[:80] + ("..." if len(ps.letter_soup_raw) > 80 else "")
        console.print(Panel(soup_display, title="Letter Soup", border_style="dim"))

    if ps.exact_words:
        console.print(f"[cyan]Exact words:[/cyan] {', '.join(ps.exact_words[:15])}")
    if ps.near_words:
        console.print(f"[cyan]Near words:[/cyan] {', '.join(ps.near_words[:10])}")
    if ps.neologisms:
        console.print(f"[cyan]Neologisms:[/cyan] {', '.join(ps.neologisms[:5])}")
    if ps.dictionary_words:
        console.print(f"[cyan]Dictionary words:[/cyan] {', '.join(ps.dictionary_words[:10])}")
    if ps.phonetic_mood:
        console.print(f"[cyan]Phonetic mood:[/cyan] {ps.phonetic_mood}")

    console.print()

    # Display atoms
    if skeleton.atoms:
        atom_lines = []
        for a in skeleton.atoms:
            tags = f" [{', '.join(a.tags)}]" if a.tags else ""
            atom_lines.append(f"  {a.category.value:>10}: {a.name}{tags}")
        console.print(Panel("\n".join(atom_lines), title="Story Atoms", border_style="green"))

    # Display spread
    if skeleton.spread_positions:
        spread_lines = [f"  {pos}: {atom}" for pos, atom in skeleton.spread_positions.items()]
        console.print(Panel("\n".join(spread_lines), title="Spread Positions", border_style="magenta"))

    # Display beats
    if skeleton.beats:
        beat_text = " → ".join(skeleton.beats)
        console.print(Panel(beat_text, title="Narrative Beats", border_style="yellow"))

    # Display stats
    stats = skeleton.stats
    info = Text()
    info.append(f"Tone: {skeleton.tone}\n")
    info.append(f"Themes: {', '.join(skeleton.theme_tags) if skeleton.theme_tags else 'none'}\n")
    info.append(f"Coherence: {stats.coherence_score:.4f}\n")
    info.append(f"Engine: {stats.engine}\n")
    if stats.violations:
        info.append(f"Violations: {', '.join(stats.violations)}")
    console.print(Panel(info, title="Stats", border_style="blue"))

    # Optional LLM synthesis
    if not args.no_llm and args.provider:
        api_key = _resolve_api_key(args.provider, args.api_key)
        if api_key:
            from dunedog.llm.provider import create_provider
            from dunedog.llm.synthesizer import StorySynthesizer
            from dunedog.models.config import LLMConfig

            provider = create_provider(args.provider, api_key=api_key, model=args.model or "")
            llm_config = LLMConfig(provider=args.provider, story_lines=20)
            synthesizer = StorySynthesizer(provider, llm_config)

            console.print("\n[bold]Synthesizing story...[/bold]")
            stories = asyncio.run(synthesizer.synthesize([skeleton]))
            for story in stories:
                console.print(Panel(Text(story.content), title=Text(story.title), border_style="blue"))


def cmd_soup(args: argparse.Namespace) -> None:
    """Generate letter soup only."""
    from dunedog.utils.seed_manager import SeedManager
    from dunedog.chaos.letter_soup import LetterSoupGenerator

    seed_mgr = SeedManager(args.seed)
    gen = LetterSoupGenerator()

    for i in range(args.count):
        rng = seed_mgr.child_rng(f"soup_{i}")
        result = gen.generate_and_parse(args.length, rng)

        console.print(Panel(result.raw_soup, title=f"Soup {i + 1}", border_style="dim"))
        if result.exact_words:
            console.print(f"  [cyan]Words found:[/cyan] {', '.join(result.exact_words[:20])}")
        if result.near_words:
            near = [f"{frag}→{match}" for frag, match, _ in result.near_words[:10]]
            console.print(f"  [cyan]Near words:[/cyan] {', '.join(near)}")
        if result.neologisms:
            console.print(f"  [cyan]Neologisms:[/cyan] {', '.join(n.text for n in result.neologisms[:5])}")
        if result.phonetic_mood:
            console.print(f"  [cyan]Mood:[/cyan] {result.phonetic_mood}")
        console.print()


def cmd_evolve(args: argparse.Namespace) -> None:
    """Evolve existing skeletons."""
    from dunedog.utils.seed_manager import SeedManager
    from dunedog.output.exporter import load_skeletons, to_json
    from dunedog.models.config import EvolutionConfig

    skeletons = load_skeletons(args.input)
    console.print(f"Loaded {len(skeletons)} skeletons from {args.input}")

    try:
        from dunedog.engines.evolutionary import StoryEvolutionEngine
    except ImportError:
        console.print("[red]Evolutionary engine not available.[/red]")
        return

    seed_mgr = SeedManager(args.seed)
    rng = seed_mgr.child_rng("evolve")

    evo_config = EvolutionConfig(
        enabled=True,
        generations=args.generations,
        population_size=len(skeletons),
    )

    engine = StoryEvolutionEngine()
    console.print(f"Evolving for {args.generations} generations...")
    result = engine.evolve(skeletons, evo_config, rng)

    console.print(f"[green]Evolution complete. Best fitness: {result.fitness_history[-1]:.4f}[/green]")
    to_json(result.population, args.output)
    console.print(f"Evolved skeletons saved to {args.output}")


def cmd_setup(_args: argparse.Namespace) -> None:
    """Download NLTK data and verify setup."""
    import nltk

    packages = [
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "wordnet",
        "words",
        "omw-1.4",
    ]

    console.print("[bold]Downloading NLTK data...[/bold]")
    for pkg in packages:
        try:
            console.print(f"  Downloading {pkg}...")
            nltk.download(pkg, quiet=True)
        except Exception as e:
            console.print(f"  [yellow]Warning: could not download {pkg}: {e}[/yellow]")

    # Verify word list
    from pathlib import Path
    data_dir = Path(__file__).resolve().parents[2] / "data"
    word_path = data_dir / "english_words.txt"
    if word_path.exists():
        with open(word_path) as f:
            count = sum(1 for _ in f)
        console.print(f"  [green]Word list: {count:,} words[/green]")
    else:
        console.print("  [yellow]Warning: english_words.txt not found[/yellow]")

    console.print("[bold green]Setup complete![/bold green]")


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "generate": cmd_generate,
        "demo": cmd_demo,
        "soup": cmd_soup,
        "evolve": cmd_evolve,
        "setup": cmd_setup,
    }

    cmd_func = commands.get(args.command)
    if not cmd_func:
        parser.print_help()
        sys.exit(1)

    try:
        cmd_func(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
