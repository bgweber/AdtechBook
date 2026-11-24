-- PDF only: let long inline code break across lines after = _ / . : , -
-- so a token like `match_type=probabilistic` can't run into the margin
-- (KDP rejects text outside the margins). The text is unchanged; TeX just
-- gets places where it may break the line if it has to.
local MIN = 18  -- shorter code fits on any line

local function esc(c)
  local map = {
    ["\\"] = "\\textbackslash{}", ["{"] = "\\{", ["}"] = "\\}", ["_"] = "\\_",
    ["#"] = "\\#", ["$"] = "\\$", ["%"] = "\\%", ["&"] = "\\&",
    ["~"] = "\\textasciitilde{}", ["^"] = "\\textasciicircum{}",
  }
  return map[c] or c
end

function Code(el)
  if not FORMAT:match("latex") then return nil end
  if #el.text < MIN or #el.classes > 0 then return nil end
  local out = {}
  for c in el.text:gmatch(".") do
    out[#out + 1] = esc(c)
    if c:match("[=_/%.:,%-]") then out[#out + 1] = "\\allowbreak{}" end
  end
  return pandoc.RawInline("latex", "\\texttt{" .. table.concat(out) .. "}")
end
