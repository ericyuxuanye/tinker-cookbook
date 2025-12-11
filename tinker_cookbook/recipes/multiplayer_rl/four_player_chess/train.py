import asyncio
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.multiplayer_rl.four_player_chess.env import FourPlayerChessDatasetBuilder
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    model_name: str = "openai/gpt-oss-20b"
    renderer_name: str | None = None
    batch_size: int = 512
    num_train_datapoints: int = 131072
    num_test_datapoints: int = 128
    learning_rate: float = 3e-5
    max_tokens: int = 512  # Enough for reasoning + move output
    eval_every: int = 5
    save_every: int = 20
    wandb_project: str | None = None
    wandb_name: str | None = None
    log_path: str | None = None


def build_config(cli_config: CLIConfig) -> train.Config:
    model_name = cli_config.model_name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"{model_name}-four-player-chess-{cli_config.batch_size}batch-{cli_config.learning_rate}lr-{date_and_time}"
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/four-player-chess/{run_name}"
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    dataset_builder = FourPlayerChessDatasetBuilder(
        batch_size=cli_config.batch_size,
        model_name=model_name,
        num_train_datapoints=cli_config.num_train_datapoints,
        num_test_datapoints=cli_config.num_test_datapoints,
        renderer_name=renderer_name,
    )

    return train.Config(
        model_name=model_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
    )


def main():
    cli_config = chz.entrypoint(CLIConfig)
    config = build_config(cli_config)
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
