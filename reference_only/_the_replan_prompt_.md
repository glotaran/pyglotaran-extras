We're looking to create a pyglotaran kinetic matrix visualizer along the lines of what was implemented before in the form of kinetic-scheme-visualizer (provided in `reference_only\kinetic-scheme-visualizer`). That implementation was created a few years ago, and we wish to create a more modern, more polished implementation.

So we have analyzed the old implementation, to find out what was good about it, what could be improved, and what should be done completely differently. The results of these analyses are available under:
reference_only\model-matrix-visualizer-analysis.md
reference_only\model-matrix-visualizer-reimplementation-plan.md

In addition to this, a preliminary plan has been created for reimplementation (this was created by Claude Code Opus 4.6 in an earlier turn), and this is available here:
reference_only\curried-coalescing-manatee.md (READ THIS PLAN FIRST)

The goal for this turn is to re-plan taking the additional analyses into account now that we are also running from within the pyglotaran-extras codebase. 

For the initial re-implementation we've decided to with a static layout (layouting) algorithm, and see if that is sufficient, before looking into the possibility for the user to interactively drag around nodes. We can also imagine a middle ground where the layouting is automatic, but the user can provide some additional directives to give additional weight to the placement and/or ordering of nodes (like node A should be to the left of node B, but B should be to the right of node C, or horizontal_layout_preference:A|C|B. 
Special thought should be given to the "decay to the ground state" - currently this is an arrow going straight down into empty space, but ideally the user should be able to "turn on" visualizing the ground state as a thick black bar. Either as a shared ground state between all models (so all the way at the bottom) or a ground state per megacomplex matrix (so a thinner, more narrow, bar just below the nodes representing the megacomplex)
A reason why static layouting might not be sufficient is given by: reference_only\why_interactive_dragging_of_nodes_is_Needed_in_cytoscape.md

Example models to test against are available under: `reference_only\kinetic-scheme-visualizer\example`

Do not edit content in the "reference_only" folder - it is there for reference only.

Ask as many clarifying questions as needed to drill down exactly what the user wants. 

Create a new, fresh plan to re-implement the kinetic-scheme-visualizer inspection module in the pyglotaran-extras.