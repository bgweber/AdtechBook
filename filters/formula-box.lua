-- Wrap display equations in a shaded box, so formulas stand apart from body text.
-- PDF: a tcolorbox (formulabox, defined in the header). HTML: <div class="formula">.
local function is_display_math_para(el)
  local math = nil
  for _, item in ipairs(el.content) do
    if item.t == "Math" and item.mathtype == "DisplayMath" and math == nil then
      math = item
    elseif item.t ~= "Space" and item.t ~= "SoftBreak" then
      return false
    end
  end
  return math ~= nil
end

function Para(el)
  if not is_display_math_para(el) then return nil end
  if FORMAT:match("latex") then
    return {
      pandoc.RawBlock("latex", "\\begin{formulabox}"),
      el,
      pandoc.RawBlock("latex", "\\end{formulabox}")
    }
  elseif FORMAT:match("html") then
    return pandoc.Div({el}, pandoc.Attr("", {"formula"}))
  end
end
