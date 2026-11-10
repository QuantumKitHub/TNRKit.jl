"""
Return the value of a given contractible network contracted using a given an environment.

The corners are given in an array of size 4, in the order northwest, northeast, southeast, southwest.
The edges are given in an array of size 4, in the order north, east, south, west.
"""

function network_value(T, corners, edges)
    return _contract_site(T, corners, edges) * _contract_corners(corners) /
        _contract_vertical_edges(corners, edges) / _contract_horizontal_edges(corners, edges)
end

function _contract_site(T, corners, edges)
    return @tensor opt = true edges[4][χ_WSW D_W; χ_WNW] *
        corners[1][χ_WNW; χ_NNW] *
        edges[1][χ_NNW D_N; χ_NNE] *
        corners[2][χ_NNE; χ_ENE] *
        edges[2][χ_ENE D_E; χ_ESE] *
        corners[3][χ_ESE; χ_SSE] *
        edges[3][χ_SSE D_S; χ_SSW] *
        corners[4][χ_SSW; χ_WSW] *
        T[D_W D_S; D_N D_E]
end

function _contract_corners(corners)
    return @tensor corners[1][1; 2] * corners[2][2; 3] * corners[3][3; 4] * corners[4][4; 1]
end

function _contract_vertical_edges(corners, edges)
    return @tensor opt = true corners[1][χ_NW; χ_N] *
        corners[2][χ_N; χ_NE] *
        edges[2][χ_NE Dh; χ_SE] *
        corners[3][χ_SE; χ_S] *
        corners[4][χ_S; χ_SW] *
        edges[4][χ_SW Dh; χ_NW]
end


function _contract_horizontal_edges(corners, edges)
    return @tensor opt = true corners[1][χ_W; χ_NW] *
        edges[1][χ_NW Dv; χ_NE] *
        corners[2][χ_NE; χ_E] *
        corners[3][χ_E; χ_SE] *
        edges[3][χ_SE Dv; χ_SW] *
        corners[4][χ_SW; χ_W]
end
