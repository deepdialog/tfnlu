from pyecharts import options as opts
from pyecharts.charts import Graph


def draw_dp(text, mat_rel_type, file='dp.html'):
    n_rel = 1 + len(text)
    nodes_data = []
    for i, x in enumerate(['H'] + list(text) + ['T']):
        nodes_data.append(opts.GraphNode(
            name=f'{x}_{i}',
            x=i * 50,
            y=100,
            value=x,
            symbol_size=10))

    links_data = []

    for i in range(n_rel):
        for j in range(n_rel):
            if mat_rel_type[i][j]:
                links_data.append(opts.GraphLink(
                    source=i, target=j,
                    value=str(mat_rel_type[i][j])
                ))
    (
        Graph()
        .add(
            "",
            nodes_data,
            links_data,
            label_opts=opts.LabelOpts(
                is_show=True, formatter="{c}"
            ),
            linestyle_opts=opts.LineStyleOpts(
                width=0.5, curve=0.5, opacity=0.7),
            edge_label=opts.LabelOpts(
                is_show=True, position="middle", formatter="{c}",
                font_size=8
            ),
            layout="none",
            is_roam=True,
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="依存")
        )
        .render(file)
    )
