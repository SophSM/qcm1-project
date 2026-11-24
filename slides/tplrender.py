"""Render a template slide layout's decorative shapes (rects, ellipses, connectors) with PIL."""
from lxml import etree
from PIL import Image, ImageDraw
NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
SCHEME={'dk1':'0E2A47','lt1':'FFFFFF','dk2':'869FB2','lt2':'FFFFFF'}
K=192/914400  # px per EMU (1920 px over 10 in)
def color(el):
    if el is None: return None
    if el.find('a:noFill',NS) is not None: return None
    sf=el.find('a:solidFill',NS)
    if sf is None: return 'inherit'
    c=sf[0]; v=c.get('val'); return '#'+(SCHEME.get(v,v) if c.tag.endswith('schemeClr') else v)
def draw_layout(layout_el, im):
    d=ImageDraw.Draw(im)
    tree=layout_el.find('.//p:cSld/p:spTree',NS)
    def walk(node):
        for ch in node:
            n=etree.QName(ch).localname
            if n=='grpSp':
                g=ch.find('p:grpSpPr/a:xfrm',NS)
                if g is not None:
                    o=g.find('a:off',NS);co=g.find('a:chOff',NS);e=g.find('a:ext',NS);ce=g.find('a:chExt',NS)
                    if (o.get('x'),o.get('y'),e.get('cx'),e.get('cy'))!=(co.get('x'),co.get('y'),ce.get('cx'),ce.get('cy')): print('WARN group transform not identity')
                yield from walk(ch)
            elif n in ('sp','cxnSp'): yield ch
    for sp in walk(tree):
        tag=etree.QName(sp).localname
        if sp.find('.//p:ph',NS) is not None: continue
        x=sp.find('.//a:xfrm',NS)
        if x is None: continue
        off=x.find('a:off',NS); ext=x.find('a:ext',NS)
        X,Y=int(off.get('x'))*K,int(off.get('y'))*K; W,H=int(ext.get('cx'))*K,int(ext.get('cy'))*K
        sppr=sp.find('p:spPr',NS); geom=sppr.find('a:prstGeom',NS); g=geom.get('prst') if geom is not None else 'rect'
        ln=sppr.find('a:ln',NS); lw=max(1,round(int(ln.get('w','9525'))*K)) if ln is not None else 1
        lc=color(ln) if ln is not None else None
        fc=color(sppr)
        if fc=='inherit': fc=None
        if lc=='inherit': lc='#0E2A47'
        if tag=='cxnSp' or 'Connector' in g or g=='line':
            x1,y1,x2,y2=X,Y,X+W,Y+H
            fh=x.get('flipH')=='1'; fv=x.get('flipV')=='1'
            if x.get('rot') in ('10800000',): fh,fv=not fh,not fv
            if fh: x1,x2=x2,x1
            if fv: y1,y2=y2,y1
            d.line([(x1,y1),(x2,y2)],fill=lc or '#0E2A47',width=lw); continue
        box=[X,Y,X+W,Y+H]
        if g=='ellipse': d.ellipse(box,fill=fc,outline=lc,width=lw)
        else: d.rectangle(box,fill=fc,outline=lc,width=lw)
