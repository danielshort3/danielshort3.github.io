"""Build the native UFO Tableau redesign while preserving its packaged extract.

Uses only Python's standard library. Pass --source for an archived source TWBX;
the default is the retained original source under design/tableau/sources. Tableau must render and
save the result before it is treated as a publication-ready workbook.
"""

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import uuid
import xml.etree.ElementTree as ET
import zipfile


SOURCE_URL = "https://public.tableau.com/workbooks/UFO_Sightings_16769494135040.twb"
SOURCE_SHA256 = "8180d5d5d71ef81fb33feb396228859582a912908455645445e56b8186630240"
USER_NS = "http://www.tableausoftware.com/xml/user"
ET.register_namespace("user", USER_NS)
BLUE = "#0877F9"
INK = "#10254F"
MUTED = "#526586"
CANVAS = "#F5F8FC"
WIDTH, HEIGHT = 1400, 1000
STATES = dict(item.split(":", 1) for item in (
  "al:Alabama|az:Arizona|ar:Arkansas|ca:California|co:Colorado|ct:Connecticut|"
  "de:Delaware|fl:Florida|ga:Georgia|id:Idaho|il:Illinois|in:Indiana|ia:Iowa|"
  "ks:Kansas|ky:Kentucky|la:Louisiana|me:Maine|md:Maryland|ma:Massachusetts|"
  "mi:Michigan|mn:Minnesota|ms:Mississippi|mo:Missouri|mt:Montana|ne:Nebraska|"
  "nv:Nevada|nh:New Hampshire|nj:New Jersey|nm:New Mexico|ny:New York|"
  "nc:North Carolina|nd:North Dakota|oh:Ohio|ok:Oklahoma|or:Oregon|"
  "pa:Pennsylvania|ri:Rhode Island|sc:South Carolina|sd:South Dakota|"
  "tn:Tennessee|tx:Texas|ut:Utah|vt:Vermont|va:Virginia|wa:Washington|"
  "wv:West Virginia|wi:Wisconsin|wy:Wyoming"
).split("|"))


def sub(parent, tag, **attrs):
  return ET.SubElement(parent, tag, {key.replace("_", "-"): str(value) for key, value in attrs.items()})


def remove(parent, path):
  for node in list(parent.findall(path)):
    parent.remove(node)


def style_rule(style, element, values, **attributes):
  rule = sub(style, "style-rule", element=element)
  for key, value in values.items():
    sub(rule, "format", attr=key, value=value, **attributes)
  return rule


def sheet_style(sheet):
  table = sheet.find("table")
  style = table.find("style")
  if style is None:
    style = ET.Element("style")
    table.insert(1, style)
  style_rule(style, "worksheet", {"font-family": "Segoe UI", "font-size": "12", "color": INK, "background-color": "#FFFFFF"})
  style_rule(style, "worksheet", {"display-field-labels": "false"}, scope="rows")
  style_rule(style, "worksheet", {"display-field-labels": "false"}, scope="cols")
  style_rule(style, "header", {"font-family": "Segoe UI", "font-size": "12", "color": INK})
  style_rule(style, "gridline", {"line-visibility": "off"})
  style_rule(style, "zeroline", {"line-visibility": "off"})
  for pane in sheet.findall(".//panes/pane"):
    ps = pane.find("style")
    if ps is None:
      ps = sub(pane, "style")
    style_rule(ps, "mark", {"mark-color": BLUE, "font-family": "Segoe UI", "font-size": "12", "mark-labels-show": "true"})


def build(source_bytes):
  source_zip = zipfile.ZipFile(io.BytesIO(source_bytes))
  names = source_zip.namelist()
  workbook_name = next(name for name in names if name.lower().endswith(".twb"))
  root = ET.fromstring(source_zip.read(workbook_name))
  ds = root.find("./datasources/datasource")
  ds_name = ds.get("name")
  prefix = f"[{ds_name}]."
  worksheets = root.find("worksheets")
  originals = {sheet.get("name"): copy.deepcopy(sheet) for sheet in worksheets}
  count_name = next(node.get("name") for node in originals["Count by State"].findall(".//column-instance") if node.get("derivation") == "Count")
  count = prefix + count_name
  idx = "[usr:Calculation_1626362480988282880:qk]"
  all_names = list(originals) + ["UFO Reports KPI", "UFO Peak Month KPI", "UFO Evening KPI", "UFO Shape Completeness", "UFO Reports by Hour"]
  custom_columns = {}

  def add_calculation(name, caption, formula, datatype="real", role="measure", field_type="quantitative", number_format=None):
    column = ET.Element("column", {"name": f"[{name}]", "caption": caption, "datatype": datatype, "role": role, "type": field_type})
    if number_format:
      column.set("default-format", number_format)
    sub(column, "calculation", **{"class": "tableau", "formula": formula})
    first_group = next((i for i, n in enumerate(ds) if n.tag in ("column-instance", "group", "extract", "layout", "semantic-values")), len(ds))
    ds.insert(first_group, column)
    custom_columns[name] = column

  add_calculation("UFO Evening Share", "Evening reports", "SUM(IF DATEPART('hour', [datetime]) >= 18 THEN 1 ELSE 0 END) / COUNT([datetime])", number_format="p0.0%")
  add_calculation("UFO Shape Missing", "Unknown or missing shape", "SUM(IF ISNULL([shape]) OR [shape] = 'unknown' THEN 1 ELSE 0 END)", "integer", number_format="#,##0")
  add_calculation("UFO Month Name", "Month", "DATENAME('month', [datetime])", "string", "dimension", "nominal")
  add_calculation("UFO Shape Label", "Shape", "IF ISNULL([shape]) THEN 'Not recorded' ELSE UPPER(LEFT([shape], 1)) + MID([shape], 2) END", "string", "dimension", "nominal")
  add_calculation("UFO City Label", "City, state", "UPPER(LEFT([city], 1)) + MID([city], 2) + ', ' + UPPER([state])", "string", "dimension", "nominal")
  contiguous_states = ", ".join(f"'{state}'" for state in sorted(STATES))
  add_calculation("UFO Geography Scope", "Contiguous United States", f"[country] = 'us' AND [state] IN ({contiguous_states})", "boolean", "dimension", "nominal")
  add_calculation("UFO Month Share", "Share of reports", "COUNT([datetime]) / WINDOW_SUM(COUNT([datetime]))", number_format="p0.0%")

  def add_dependency(sheet, name, aggregate=False):
    deps = sheet.find("./table/view/datasource-dependencies")
    column = custom_columns[name]
    if deps.find(f"column[@name='[{name}]']") is None:
      first_instance = next((i for i,n in enumerate(deps) if n.tag == "column-instance"), len(deps))
      deps.insert(first_instance, copy.deepcopy(column))
    field_type = "quantitative" if aggregate else column.get("type")
    suffix = "qk" if field_type == "quantitative" else "nk"
    instance = f"[{'usr' if aggregate else 'none'}:{name}:{suffix}]"
    if deps.find(f"column-instance[@name='{instance}']") is None:
      sub(deps, "column-instance", column=f"[{name}]", derivation="User" if aggregate else "None", name=instance, pivot="key", type=field_type)
    return prefix + instance

  def context_filters(sheet):
    view = sheet.find("./table/view")
    remove(view, "filter")
    remove(view, "slices")
    additions = []
    geo = add_dependency(sheet, "UFO Geography Scope")
    geo_filter = ET.Element("filter", {"class": "categorical", "column": geo, "context": "true"})
    sub(geo_filter, "groupfilter", function="member", level=geo[len(prefix):], member="true", **{f"{{{USER_NS}}}ui-enumeration": "inclusive"})
    additions.append(geo_filter)
    for field, group, member in [("[yr:datetime:ok]", 1, "2013"), ("[none:state:nk]", 2, None), ("[none:UFO Shape Label:nk]", 3, None)]:
      filt = ET.Element("filter", {"class": "categorical", "column": prefix + field, "filter-group": str(group), "context": "true"})
      attrs = {"function": "member" if member else "level-members", "level": field, f"{{{USER_NS}}}ui-domain": "relevant", f"{{{USER_NS}}}ui-enumeration": "inclusive" if member else "all", f"{{{USER_NS}}}ui-marker": "enumerate"}
      if member:
        attrs["member"] = member
      ET.SubElement(filt, "groupfilter", attrs)
      additions.append(filt)
    add_dependency(sheet, "UFO Shape Label")
    for filt in additions:
      view.insert(next((i for i,n in enumerate(view) if n.tag in ("manual-sort", "computed-sort", "shelf-sorts", "aggregation")), len(view)), filt)
    for node in sheet.findall(".//column[@name='[state]']"):
      remove(node, "aliases")
      aliases = ET.Element("aliases")
      for code, name in STATES.items():
        sub(aliases, "alias", key=f'"{code}"', value=name)
      sem = node.find("semantic-values")
      node.insert(list(node).index(sem) if sem is not None else 0, aliases)
    # Every sheet receives the same actual state/year/shape fields and aliases.
    deps = sheet.find("./table/view/datasource-dependencies")
    for original_name in ("[country]", "[state]", "[datetime]", "[shape]"):
      if deps.find(f"column[@name='{original_name}']") is None:
        deps.insert(0, copy.deepcopy(ds.find(f"column[@name='{original_name}']")))
    for original_name, derivation, instance, field_type in [("[state]", "None", "[none:state:nk]", "nominal"), ("[country]", "None", "[none:country:nk]", "nominal"), ("[datetime]", "Year", "[yr:datetime:ok]", "ordinal")]:
      if deps.find(f"column-instance[@name='{instance}']") is None:
        sub(deps, "column-instance", column=original_name, derivation=derivation, name=instance, pivot="key", type=field_type)

  def index_filter(sheet, top):
    view = sheet.find("./table/view")
    deps = view.find("datasource-dependencies")
    original_deps = originals["Count by State"].find("./table/view/datasource-dependencies")
    for tag, name in [("column", "[Calculation_1626362480988282880]"), ("column-instance", idx)]:
      if deps.find(f"{tag}[@name='{name}']") is None:
        deps.append(copy.deepcopy(original_deps.find(f"{tag}[@name='{name}']")))
    filt = ET.Element("filter", {"class": "quantitative", "column": prefix + idx, "included-values": "in-range"})
    sub(filt, "min").text = "1"
    sub(filt, "max").text = str(top)
    view.insert(next((i for i,n in enumerate(view) if n.tag in ("manual-sort", "computed-sort", "shelf-sorts", "aggregation")), len(view)), filt)

  def horizontal(sheet, dimension, top=None):
    table = sheet.find("table")
    table.find("rows").text = dimension
    table.find("cols").text = count
    view = table.find("view")
    remove(view, "shelf-sorts")
    remove(view, "computed-sort")
    sorts = ET.Element("shelf-sorts")
    sub(sorts, "shelf-sort-v2", dimension_to_sort=dimension, direction="DESC", is_on_innermost_dimension="true", measure_to_sort_by=count, shelf="rows")
    view.insert(list(view).index(view.find("aggregation")), sorts)
    pane = table.find("./panes/pane")
    pane.find("mark").set("class", "Bar")
    enc = pane.find("encodings")
    remove(enc, "color")
    if enc.find("text") is None:
      sub(enc, "text", column=count)
    for rule in table.findall("./style/style-rule[@element='axis']"):
      table.find("style").remove(rule)
    if top:
      index_filter(sheet, top)
    sheet_style(sheet)

  for sheet in worksheets:
    context_filters(sheet)
    sheet_style(sheet)
  for sheet_name in ("Count by State Map", "Count by City Map", "Month/Hour (Military Time) Heatmap"):
    styled_sheet = worksheets.find(f"worksheet[@name='{sheet_name}']")
    style = styled_sheet.find("./table/style")
    for encoding in style.findall(".//encoding[@attr='color']"):
      encoding.set("palette", "blue_10_0")
    rule = sub(style, "style-rule", element="mark")
    sub(rule, "encoding", attr="color", field=count, palette="blue_10_0", type="interpolated")
    if "Map" in sheet_name:
      style_rule(style, "map", {"washout": "1.0"})
  state = worksheets.find("worksheet[@name='Count by State']")
  horizontal(state, prefix + "[none:state:nk]", 5)
  shapes = worksheets.find("worksheet[@name='Count by Shape']")
  horizontal(shapes, add_dependency(shapes, "UFO Shape Label"), 5)
  city = worksheets.find("worksheet[@name='Count by City']")
  horizontal(city, add_dependency(city, "UFO City Label"), 10)
  # No hidden minimum-volume cutoff remains on the city map.
  for shape_name in ("Shape Prevailance", "Shape Prevailance Table"):
    sheet = worksheets.find(f"worksheet[@name='{shape_name}']")
    shape_field = add_dependency(sheet, "UFO Shape Label")
    for node in sheet.iter():
      for key,value in list(node.attrib.items()):
        if value == prefix + "[none:shape:nk]":
          node.set(key, shape_field)
      if node.text:
        node.text = node.text.replace(prefix + "[none:shape:nk]", shape_field)
  heat = worksheets.find("worksheet[@name='Month/Hour (Military Time) Heatmap']")
  dictionary = heat.find(".//manual-sort/dictionary")
  dictionary.clear()
  for hour in range(24):
    sub(dictionary, "bucket").text = str(hour)
  for rule in heat.findall("./table/style/style-rule[@element='cell']") + heat.findall("./table/style/style-rule[@element='header']"):
    heat.find("./table/style").remove(rule)
  style_rule(heat.find("./table/style"), "header", {"font-size": "10"})
  heat.find("./table/panes/pane/mark").set("class", "Square")
  heat.find("./table/panes/pane/style/style-rule[@element='mark']/format[@attr='size']").set("value", "1.0")

  def clone(source, name):
    sheet = copy.deepcopy(worksheets.find(f"worksheet[@name='{source}']"))
    sheet.set("name", name)
    remove(sheet, "repository-location")
    sheet.find("simple-id").set("uuid", "{" + str(uuid.uuid5(uuid.NAMESPACE_URL, "ufo-redesign/" + name)).upper() + "}")
    context_filters(sheet)
    worksheets.append(sheet)
    return sheet

  def text_sheet(sheet, labels, fields, dimension=None):
    table = sheet.find("table")
    table.find("rows").text = ""
    table.find("cols").text = ""
    view = table.find("view")
    remove(view, "shelf-sorts")
    remove(view, "computed-sort")
    pane = table.find("./panes/pane")
    pane.find("mark").set("class", "Text")
    enc = pane.find("encodings")
    enc.clear()
    for field in fields:
      sub(enc, "text", column=field)
    if dimension:
      sub(enc, "lod", column=dimension)
    remove(pane, "customized-label")
    label = ET.Element("customized-label")
    ft = sub(label, "formatted-text")
    for text, size, bold, color in labels:
      sub(ft, "run", fontname="Segoe UI", fontsize=size, bold=str(bold).lower(), fontcolor=color).text = text
    pane.insert(list(pane).index(pane.find("style")), label)
    style_rule(table.find("style"), "pane", {"background-color": "#FFFFFF"})
    style_rule(pane.find("style"), "mark", {"mark-labels-cull": "false", "mark-labels-show": "true", "text-align": "left", "vertical-align": "center"})

  reports = clone("Count by State", "UFO Reports KPI")
  text_sheet(reports, [("Reports\n", 13, False, INK), (f"<{count}>\n", 32, True, INK), ("Current selection · contiguous U.S.", 11, False, MUTED)], [count])
  peak = clone("Count by State", "UFO Peak Month KPI")
  month = add_dependency(peak, "UFO Month Name")
  text_sheet(peak, [("Peak month\n", 13, False, INK), (f"<{month}>\n", 30, True, INK), (f"<{count}> reports", 11, False, MUTED)], [count, month], month)
  # INDEX() inherited from the source explicitly computes down Rows. Keep the
  # month dimension on Rows so the top-one table calculation has that address.
  peak.find("./table/rows").text = month
  style_rule(peak.find("./table/style"), "header", {"display": "false"}, field=month, scope="rows")
  peak_view = peak.find("./table/view")
  sort = ET.Element("computed-sort", {"column": month, "direction": "DESC", "using": count})
  peak_view.insert(list(peak_view).index(peak_view.find("aggregation")), sort)
  index_filter(peak, 1)
  evening = clone("Count by State", "UFO Evening KPI")
  share = add_dependency(evening, "UFO Evening Share", True)
  text_sheet(evening, [("Evening reports\n", 13, False, INK), (f"<{share}>\n", 32, True, INK), ("Recorded hour 18:00–23:59", 11, False, MUTED)], [share])
  completeness = clone("Count by State", "UFO Shape Completeness")
  missing = add_dependency(completeness, "UFO Shape Missing", True)
  text_sheet(completeness, [(f"Unknown or missing shape: <{missing}> reports", 10, False, MUTED)], [missing])
  hourly = clone("Month/Hour (Military Time) Heatmap", "UFO Reports by Hour")
  horizontal(hourly, prefix + "[hr:datetime:ok]")
  remove(hourly.find("./table/view"), "shelf-sorts")

  dashboard = root.find("./dashboards/dashboard")
  dashboard_name = dashboard.get("name")
  remove(dashboard, "layout-options")
  remove(dashboard, "zones")
  remove(dashboard, "devicelayouts")
  dashboard.find("size").attrib = {"maxheight": str(HEIGHT), "maxwidth": str(WIDTH), "minheight": str(HEIGHT), "minwidth": str(WIDTH), "sizing-mode": "fixed"}
  dashboard_style = dashboard.find("style")
  style_rule(dashboard_style, "dashboard", {"background-color": CANVAS})
  zones = ET.Element("zones")
  dashboard.insert(list(dashboard).index(dashboard.find("simple-id")), zones)
  container = sub(zones, "zone", id=1, type_v2="layout-basic", x=0, y=0, w=100000, h=100000)
  zone_ids = {}
  next_id = 10

  def zone(x,y,w,h,name=None,kind=None,text=None,font_size=12,color=INK,bold=False,fill=None,param=None):
    nonlocal next_id
    attrs = {"id": str(next_id), "x": str(round(x/WIDTH*100000)), "y": str(round(y/HEIGHT*100000)), "w": str(round(w/WIDTH*100000)), "h": str(round(h/HEIGHT*100000))}
    next_id += 1
    if name:
      attrs.update(name=name, **{"show-title": "false"})
      if not kind:
        zone_ids[name] = attrs["id"]
    if kind:
      attrs["type-v2"] = kind
    if param:
      attrs["param"] = param
    z = ET.SubElement(container, "zone", attrs)
    if text:
      sub(sub(z, "formatted-text"), "run", fontname="Segoe UI", fontsize=font_size, fontcolor=color, bold=str(bold).lower()).text=text
    zs=sub(z,"zone-style")
    for attr,value in {"border-width":"0","border-style":"none","margin":"0","padding":"8","background-color":fill or "#FFFFFF"}.items():
      sub(zs,"format",attr=attr,value=value)
    return z

  zone(20,12,950,27,kind="text",text="DANIEL SHORT / DATA EXPLORER",font_size=10,color=MUTED,bold=True,fill=CANVAS)
  zone(20,37,1170,45,kind="text",text="UFO sighting reports",font_size=28,bold=True,fill=CANVAS)
  zone(20,81,1200,32,kind="text",text="Reported observations across the contiguous United States",font_size=13,color=MUTED,fill=CANVAS)
  for x,w,field in [(24,180,"[yr:datetime:ok]"),(218,260,"[none:state:nk]"),(490,250,"[none:UFO Shape Label:nk]")]:
    control=zone(x,119,w,58,name="Count by State Map",kind="filter",param=prefix+field,fill=CANVAS)
    control.set("mode", "dropdown")
    control.set("show-all", "true")
    control.set("values", "relevant")
  zone(770,126,610,44,kind="text",text="Select a state or shape to explore. Use Tableau Revert to reset.",font_size=10,color=MUTED,fill=CANVAS)
  zone(20,188,450,105,name="UFO Reports KPI")
  zone(475,188,450,105,name="UFO Peak Month KPI")
  zone(930,188,450,105,name="UFO Evening KPI")
  zone(20,305,840,41,kind="text",text="Reports by state",font_size=19,bold=True)
  zone(20,346,840,29,kind="text",text="Report counts · select a state to filter the dashboard",color=MUTED,font_size=11)
  zone(20,375,840,265,name="Count by State Map")
  zone(20,640,600,28,kind="text",text="Raw counts; not population-adjusted.",font_size=10,color=MUTED)
  zone(630,640,230,28,name="Count by State Map",kind="color",param=count)
  zone(875,305,505,42,kind="text",text="Top 5 states",font_size=19,bold=True)
  zone(875,347,505,321,name="Count by State")
  zone(20,684,840,41,kind="text",text="Reports by month and hour",font_size=19,bold=True)
  zone(20,725,840,28,kind="text",text="Recorded observation time · hours ordered midnight to midnight",font_size=11,color=MUTED)
  zone(20,753,840,198,name="Month/Hour (Military Time) Heatmap")
  zone(875,684,505,41,kind="text",text="Most reported shapes",font_size=19,bold=True)
  zone(875,725,505,28,kind="text",text="Top 5 · all shapes included in totals",font_size=11,color=MUTED)
  zone(875,753,505,170,name="Count by Shape")
  zone(875,923,505,28,name="UFO Shape Completeness")
  zone(20,964,1360,30,kind="text",text="How to read this: Counts describe submitted reports, not verified events or incidence rates.  Source: historical NUFORC-derived workbook.",font_size=10,color=MUTED)

  devices = ET.Element("devicelayouts")
  dashboard.insert(list(dashboard).index(dashboard.find("simple-id")), devices)
  phone=sub(devices,"devicelayout",name="Phone")
  phone_height=2450
  sub(phone,"size",maxheight=phone_height,minheight=phone_height,sizing_mode="vscroll")
  pzones=sub(phone,"zones")
  pc=sub(pzones,"zone",id=1,type_v2="layout-basic",x=0,y=0,w=100000,h=100000)
  py=10
  for control in container.findall("zone[@type-v2='filter']"):
    phone_control=copy.deepcopy(control)
    phone_control.attrib.update(x="2000",y=str(round(py/phone_height*100000)),w="96000",h=str(round(62/phone_height*100000)))
    pc.append(phone_control)
    py+=70
  for name,ph in [("UFO Reports KPI",125),("UFO Peak Month KPI",125),("UFO Evening KPI",125),("Count by State",280),("Count by State Map",260),("UFO Reports by Hour",520),("Count by Shape",260),("UFO Shape Completeness",50)]:
    ident=zone_ids.get(name,str(next_id))
    next_id += 1
    sub(pc,"zone",id=ident,name=name,x=2000,y=round(py/phone_height*100000),w=96000,h=round(ph/phone_height*100000),show_title="true")
    py += ph+12

  # Tableau's native generated filter actions (same mechanism as the source Pizza workbook).
  actions = root.find("actions")
  if actions is None:
    actions=ET.Element("actions")
    root.insert(list(root).index(worksheets),actions)
  for source_name in ["Count by State Map", "Count by State", "Count by Shape"]:
    action_id="[UFO_" + str(uuid.uuid5(uuid.NAMESPACE_URL, source_name)).replace("-", "") + "]"
    action=sub(actions,"action",caption="Explore " + source_name,name=action_id)
    sub(action,"activation",auto_clear="true",type="on-select")
    sub(action,"source",dashboard=dashboard_name,type="sheet",worksheet=source_name)
    command=sub(action,"command",command="tsc:tsl-filter")
    sub(command,"param",name="special-fields",value="all")
    sub(command,"param",name="target",value=dashboard_name)
    levels=["[UFO Shape Label]"] if source_name=="Count by Shape" else ["[country]","[state]"] if source_name=="Count by State Map" else ["[state]"]
    group_name="[Action (UFO " + source_name + ")]"
    group=ET.Element("group",{"caption":group_name[1:-1],"name":group_name,"hidden":"true","name-style":"unqualified",f"{{{USER_NS}}}auto-column":"sheet_link"})
    group_join=sub(group,"groupfilter",function="crossjoin")
    for level in levels:
      sub(group_join,"groupfilter",function="level-members",level=level)
    ds.insert(next((i for i,n in enumerate(ds) if n.tag=="group"),len(ds)),group)
    for target_sheet in worksheets:
      if target_sheet.get("name")==source_name:
        continue
      view=target_sheet.find("./table/view")
      receiver=ET.Element("filter",{"class":"categorical","column":prefix+group_name})
      filter_attrs={"function":"level-members" if len(levels)==1 else "crossjoin",f"{{{USER_NS}}}ui-action-filter":action_id,f"{{{USER_NS}}}ui-enumeration":"all",f"{{{USER_NS}}}ui-marker":"enumerate"}
      if len(levels)==1:
        filter_attrs["level"]=levels[0]
      receiver_group=ET.SubElement(receiver,"groupfilter",filter_attrs)
      if len(levels)>1:
        for level in levels:
          sub(receiver_group,"groupfilter",function="level-members",level=level)
      view.insert(next((i for i,n in enumerate(view) if n.tag in ("manual-sort","computed-sort","shelf-sorts","slices","aggregation")),len(view)),receiver)
  # Retain explicit filter slices, including generated action receiver fields.
  for sheet in worksheets:
    view=sheet.find("./table/view")
    remove(view,"slices")
    slices=ET.Element("slices")
    for filt in view.findall("filter"):
      sub(slices,"column").text=filt.get("column")
    view.insert(list(view).index(view.find("aggregation")),slices)

  windows=root.find("windows")
  for name in all_names:
    win=windows.find(f"window[@class='worksheet'][@name='{name}']")
    if win is None:
      win=sub(windows,"window",**{"class":"worksheet","name":name})
      sub(win,"cards")
    viewport=win.find("viewpoint")
    if viewport is None:
      viewport=sub(win,"viewpoint")
    for z in viewport.findall("zoom"):
      viewport.remove(z)
    viewport.insert(0,ET.Element("zoom",{"type":"entire-view"}))
  dwin=windows.find(f"window[@class='dashboard'][@name='{dashboard_name}']")
  vps=dwin.find("viewpoints")
  vps.clear()
  for name in all_names:
    sub(sub(vps,"viewpoint",name=name),"zoom",type="entire-view")
  # Original workbook identifier, dashboard identifier, and Hyper bytes are retained.
  ET.indent(root, space="  ")
  xml=ET.tostring(root,encoding="utf-8",xml_declaration=True)
  output=io.BytesIO()
  with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as target:
    for name in names:
      info=zipfile.ZipInfo(name,date_time=(2026,9,11,0,0,0))
      info.compress_type=zipfile.ZIP_DEFLATED
      target.writestr(info,xml if name==workbook_name else source_zip.read(name))
  validate(root, source_zip, zipfile.ZipFile(io.BytesIO(output.getvalue())))
  manifest={"sourceUrl":SOURCE_URL,"sourceSha256":hashlib.sha256(source_bytes).hexdigest(),"workbook":root.find("repository-location").attrib,"dashboard":dashboard_name,"defaultScope":{"country":"us","year":2013,"includedStates":sorted(STATES),"expectedReports":6334},"worksheets":all_names,"extractSha256":{n:hashlib.sha256(source_zip.read(n)).hexdigest() for n in names if n.endswith(".hyper")},"validation":"Structural and extract-integrity checks passed. Native Tableau rendering and filter behavior are not yet validated."}
  return output.getvalue(), xml, manifest


def validate(root, source_zip, target_zip):
  sheets={n.get("name") for n in root.findall("./worksheets/worksheet")}
  for zone in root.findall(".//dashboard//zone"):
    if zone.get("name"):
      assert zone.get("name") in sheets, zone.get("name")
  for name in source_zip.namelist():
    if name.endswith(".hyper"):
      assert source_zip.read(name)==target_zip.read(name), "Extract changed"
  for sheet in root.findall("./worksheets/worksheet"):
    filters=sheet.findall("./table/view/filter")
    assert any(n.get("filter-group")=="1" for n in filters)
    assert any(n.get("filter-group")=="2" for n in filters)
    assert any(n.get("filter-group")=="3" for n in filters)
  heat=root.find("./worksheets/worksheet[@name='Month/Hour (Military Time) Heatmap']")
  assert [int(n.text) for n in heat.findall(".//manual-sort/dictionary/bucket")]==list(range(24))
  city=root.find("./worksheets/worksheet[@name='Count by City']/table/rows")
  assert "UFO City Label" in city.text


def check_schema(xml, source, schema_path):
  """Check Tableau's core schema with the namespaces omitted from its download.

  Compare with the original workbook because this published XSD also rejects
  the original Public export's absent trailing metadata. Metadata namespace
  attributes are accepted permissively; core TWB rules are left untouched.
  """
  from lxml import etree

  class NamespaceResolver(etree.Resolver):
    def resolve(self, url, public_id, context):
      if url.endswith("ufo-user.xsd"):
        return self.resolve_string('<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="http://www.tableausoftware.com/xml/user"><xs:attributeGroup name="UserAttributes-AG"><xs:anyAttribute namespace="http://www.tableausoftware.com/xml/user" processContents="skip"/></xs:attributeGroup></xs:schema>', context)
      if url.endswith("ufo-xml.xsd"):
        return self.resolve_string('<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="http://www.w3.org/XML/1998/namespace"><xs:attribute name="base" type="xs:anyURI"/></xs:schema>', context)
      return None

  parser = etree.XMLParser(no_network=True)
  parser.resolvers.add(NamespaceResolver())
  schema_text = schema_path.read_text(encoding="utf-8")
  schema_text = schema_text.replace('<xs:import namespace="http://www.tableausoftware.com/xml/user"/>', '<xs:import namespace="http://www.tableausoftware.com/xml/user" schemaLocation="ufo-user.xsd"/>')
  schema_text = schema_text.replace('<xs:import namespace="http://www.w3.org/XML/1998/namespace"/>', '<xs:import namespace="http://www.w3.org/XML/1998/namespace" schemaLocation="ufo-xml.xsd"/>')
  schema = etree.XMLSchema(etree.fromstring(schema_text.encode(), parser))
  with zipfile.ZipFile(io.BytesIO(source)) as archive:
    original_xml = archive.read(next(name for name in archive.namelist() if name.endswith(".twb")))
  schema.validate(etree.fromstring(original_xml))
  original_errors = {error.message for error in schema.error_log}
  passed = schema.validate(etree.fromstring(xml))
  errors = [error.message for error in schema.error_log]
  introduced = [error for error in errors if error not in original_errors]
  return {"status": "passed" if passed else "passed with inherited source deviation" if not introduced else "failed", "introducedErrors": introduced, "inheritedErrors": [error for error in errors if error in original_errors], "schema": str(schema_path), "namespaceResolution": "Supplemental XML and permissive user-metadata declarations"}


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source",type=Path,default=Path(__file__).resolve().parents[2]/"design/tableau/sources/UFO_Sightings.twbx",help="Exact original packaged TWBX (the downloaded .twb is also a ZIP)")
  parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[2]/"design/tableau/ufo-sightings-revamped.twbx")
  parser.add_argument("--schema",type=Path,help="Optional official TWB XSD (requires lxml); compare new errors with the source baseline")
  args=parser.parse_args()
  source=args.source.read_bytes()
  if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
    raise ValueError("Source is not the audited original snapshot; inspect its schema and update the pinned source hash before rebuilding.")
  twbx,twb,manifest=build(source)
  if args.schema:
    manifest["schemaValidation"] = check_schema(twb, source, args.schema)
    if manifest["schemaValidation"]["introducedErrors"]:
      raise ValueError(json.dumps(manifest["schemaValidation"], indent=2))
  args.output.parent.mkdir(parents=True,exist_ok=True)
  args.output.write_bytes(twbx)
  args.output.with_suffix(".twb").write_bytes(twb)
  args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
  print(json.dumps({"output":str(args.output),"bytes":len(twbx),"validation":manifest["validation"]},indent=2))


if __name__=="__main__":
  main()
