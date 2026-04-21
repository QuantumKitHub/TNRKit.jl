using TNRKit
using Documenter
using DocumenterCitations

bibpath = joinpath(@__DIR__, "src", "assets", "tnrkit.bib")
bib = CitationBibliography(bibpath; style = :authoryear)

makedocs(;
    sitename = "TNRKit.jl",
    pages = [
        "Home" => "index.md"
        "Library" => "lib/lib.md"
        "CFT Data" => "cft.md"
        "Finalizers" => "finalizers.md"
        "References" => "references.md"
    ],
    plugins = [bib]
)

deploydocs(; repo = "github.com/VictorVanthilt/TNRKit.jl.git", push_preview = true)
