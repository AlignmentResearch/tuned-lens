import pandas as pd
import plotly.graph_objects as go
import torch as th


def plot_stimulus_response_alignment(df: pd.DataFrame) -> go.Figure:
    layer_means = df.groupby("stimulus_layer")["sr_alignment"].mean()

    cell_grouped = df.groupby(["stimulus_layer", "token_index"])
    cell_means = cell_grouped["sr_alignment"].mean().reset_index("stimulus_layer")
    S = cell_means.index.max() + 1

    fig = go.Figure(
        [
            go.Scatter(
                x=cell_means.stimulus_layer.loc[i],
                y=cell_means.sr_alignment.loc[i] if i else layer_means,
                mode="lines+markers",
                name=f"Token {i}" if i else "All tokens",
                visible=i == 0,
            )
            for i in range(S)
        ]
    )
    fig.update_layout(
        sliders=[
            dict(
                currentvalue=dict(prefix="Token index: "),
                steps=[
                    dict(
                        args=[dict(visible=visible_mask)],
                        label=str(i) if i else "all",
                        method="restyle",
                    )
                    for i, visible_mask in enumerate(th.eye(S).bool())
                ],
            )
        ],
        title="Stimulus-response alignment by layer",
    )
    fig.update_xaxes(title="Stimulus layer")
    fig.update_yaxes(
        range=[min(0, cell_means.sr_alignment.min()), 1], title="Aitchison similarity"
    )
    return fig
