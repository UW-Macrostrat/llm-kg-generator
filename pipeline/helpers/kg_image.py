from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def _run_directory(ctx):
    """
    Find the current run output directory.
    """

    for obj, attribute in (
        (ctx.log, "run_dir"),
        (ctx.log, "root"),
        (ctx.log, "output_dir"),
    ):
        value = getattr(
            obj,
            attribute,
            None,
        )

        if value is not None:
            return Path(value)

    output = getattr(
        ctx.config,
        "output",
        None,
    )

    if output is not None:
        return Path(output)

    return Path("pipeline_output")


def save_kg_image(
    ctx,
    source_id,
    entities,
    relations,
    filename="kg.png",
    title=None,
):
    """
    Save a KG image for one relation section / sentence.

    Only supplied entities and relations are rendered.
    The caller should therefore pass only entities participating
    in relationships for that sentence.
    """

    if not relations:
        return None

    graph = nx.DiGraph()

    entity_by_id = {
        entity.id: entity
        for entity in entities
    }

    # ---------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------

    for entity in entities:
        graph.add_node(
            entity.id,
            label=(
                f"{entity.text}\n"
                f"[{entity.entity_type}]"
            ),
        )

    # ---------------------------------------------------------
    # Edges
    #
    # Cycles such as A -> B and B -> A are allowed.
    # Multiple relationship types between the same directed
    # endpoints are combined into one displayed edge label.
    # ---------------------------------------------------------

    for relation in relations:
        if (
            relation.src_id not in entity_by_id
            or relation.dst_id not in entity_by_id
        ):
            continue

        if graph.has_edge(
            relation.src_id,
            relation.dst_id,
        ):
            existing = graph[
                relation.src_id
            ][
                relation.dst_id
            ].get(
                "label",
                "",
            )

            relationship_types = {
                value
                for value in existing.split("\n")
                if value
            }

            relationship_types.add(
                relation.relationship_type
            )

            graph[
                relation.src_id
            ][
                relation.dst_id
            ][
                "label"
            ] = "\n".join(
                sorted(relationship_types)
            )

        else:
            graph.add_edge(
                relation.src_id,
                relation.dst_id,
                label=relation.relationship_type,
            )

    if not graph.edges:
        return None

    # ---------------------------------------------------------
    # Layout
    # ---------------------------------------------------------

    node_count = max(
        len(graph.nodes),
        1,
    )

    # Sentence-level graphs should usually be fairly small.
    width = min(
        20,
        max(
            9,
            node_count * 1.6,
        ),
    )

    height = min(
        16,
        max(
            7,
            node_count * 1.1,
        ),
    )

    figure, ax = plt.subplots(
        figsize=(width, height),
    )

    position = nx.spring_layout(
        graph,
        seed=42,
        k=max(
            1.0,
            2.5 / max(
                node_count ** 0.5,
                1,
            ),
        ),
        iterations=300,
    )

    # ---------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------

    labels = nx.get_node_attributes(
        graph,
        "label",
    )

    nx.draw_networkx_nodes(
        graph,
        position,
        node_size=2200,
        ax=ax,
    )

    nx.draw_networkx_labels(
        graph,
        position,
        labels=labels,
        font_size=8,
        ax=ax,
    )

    # ---------------------------------------------------------
    # Edges
    # ---------------------------------------------------------

    nx.draw_networkx_edges(
        graph,
        position,
        arrows=True,
        arrowsize=18,
        node_size=2200,
        connectionstyle="arc3,rad=0.10",
        ax=ax,
    )

    edge_labels = nx.get_edge_attributes(
        graph,
        "label",
    )

    nx.draw_networkx_edge_labels(
        graph,
        position,
        edge_labels=edge_labels,
        font_size=7,
        rotate=False,
        ax=ax,
    )

    # ---------------------------------------------------------
    # Sentence title
    # ---------------------------------------------------------

    if title:
        ax.set_title(
            title,
            fontsize=10,
            wrap=True,
            pad=20,
        )

    ax.set_axis_off()

    figure.tight_layout()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    run_dir = _run_directory(ctx)

    output = (
        run_dir
        / "sources"
        / str(source_id)
        / filename
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    ctx.log.event(
        "INFO",
        "kg_image_saved",
        stage="re",
        source_id=source_id,
        path=str(output),
        filename=filename,
        nodes=len(graph.nodes),
        edges=len(graph.edges),
    )

    return output