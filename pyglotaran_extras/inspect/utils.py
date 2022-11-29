"""Inspection utility module."""

from __future__ import annotations


def wrap_in_details_tag(
    details_content: str,
    *,
    summary_content: str | None = None,
    summary_heading_level: int | None = None,
    is_open: bool = False,
) -> str:
    """Wrap ``details_content`` in a html details tag and add summary if ``summary_content`` set.

    Parameters
    ----------
    details_content: str
        Markdown string that should be displayed when the details are expanded.
    summary_content: str | None
        Summary test that should be displayed. Defaults to None so the summary is ``Details``.
    summary_heading_level: int | None
        Level of the heading wrapping the ``summary`` if it is not None. Defaults to None
    is_open: bool
        Whether or not the details tag should be initially opened. Defaults to False

    Returns
    -------
    str
    """
    out_str = f'\n<details {"open" if is_open else ""}>\n'
    if summary_content is not None:
        out_str += "<summary>\n"
        if summary_heading_level is None:
            out_str += f"{summary_content}\n"
        else:
            # Ref.:
            # https://css-tricks.com/two-issues-styling-the-details-element-and-how-to-solve-them/
            out_str += f'<h{summary_heading_level} style="display:inline;">\n'
            out_str += f"{summary_content}\n"
            out_str += f"</h{summary_heading_level}>\n"

        out_str += "</summary>\n"

    out_str += f"\n{details_content}\n</details>"
    return out_str
