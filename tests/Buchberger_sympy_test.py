"""Tests for Groebner bases. """

import pytest

from sympy.polys.orderings import lex, grlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ

from grobnerRl.Buchberger.BuchbergerSympy import groebner, _f5b, _buchberger


@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_Buchberger_case1(algorithm):
    R, x,y = ring("x,y", QQ, lex)
    f = x**2 + 2*x*y**2
    g = x*y + 2*y**3 - 1

    assert groebner([f, g], R, algorithm) == [x, y**3 - QQ(1,2)]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_Buchberger_case2(algorithm):
    R, y,x = ring("y,x", QQ, lex)
    f = 2*x**2*y + y**2
    g = 2*x**3 + x*y - 1

    assert groebner([f, g], R, algorithm) == [y, x**3 - QQ(1,2)]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_Buchberger_case3(algorithm):
    R, x,y,z = ring("x,y,z", QQ, lex)
    f = x - z**2
    g = y - z**3

    assert groebner([f, g], R, algorithm) == [f, g]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_Buchberger_case4(algorithm):
    R, x,y = ring("x,y", QQ, grlex)
    f = x**3 - 2*x*y
    g = x**2*y + x - 2*y**2

    assert groebner([f, g], R, algorithm) == [x**2, x*y, -QQ(1,2)*x + y**2]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_Buchberger_case5(algorithm):
    R, x,y,z = ring("x,y,z", QQ, lex)
    f = -x**2 + y
    g = -x**3 + z

    assert groebner([f, g], R, algorithm) == [x**2 - y, x*y - z, x*z - y**2, y**3 - z**2]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_Buchberger_case6(algorithm):
    R, x,y,z = ring("x,y,z", QQ, grlex)
    f = -x**2 + y
    g = -x**3 + z

    assert groebner([f, g], R, algorithm) == [y**3 - z**2, x**2 - y, x*y - z, x*z - y**2]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_Buchberger_case7(algorithm):
    R, x,y,z = ring("x,y,z", QQ, lex)
    f = -x**2 + z
    g = -x**3 + y

    assert groebner([f, g], R, algorithm) == [x**2 - z, x*y - z**2, x*z - y, y**2 - z**3]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_Buchberger_case8(algorithm):
    R, x,y,z = ring("x,y,z", QQ, grlex)
    f = -x**2 + z
    g = -x**3 + y

    assert groebner([f, g], R, algorithm) == [-y**2 + z**3, x**2 - z, x*y - z**2, x*z - y]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_Buchberger_case9(algorithm):
    R, x,y,z = ring("x,y,z", QQ, lex)
    f = x - y**2
    g = -y**3 + z

    assert groebner([f, g], R, algorithm) == [x - y**2, y**3 - z]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_Buchberger_case10(algorithm):
    R, x,y,z = ring("x,y,z", QQ, grlex)
    f = x - y**2
    g = -y**3 + z

    assert groebner([f, g], R, algorithm) == [x**2 - y*z, x*y - z, -x + y**2]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_Buchberger_case11(algorithm):
    R, x,y,z = ring("x,y,z", QQ, lex)
    f = x - z**2
    g = y - z**3

    assert groebner([f, g], R, algorithm) == [x - z**2, y - z**3]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_Buchberger_case12(algorithm):
    R, x,y,z = ring("x,y,z", QQ, grlex)
    f = x - z**2
    g = y - z**3

    assert groebner([f, g], R, algorithm) == [x**2 - y*z, x*z - y, -x + z**2]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_Buchberger_case13(algorithm):
    R, x,y,z = ring("x,y,z", QQ, lex)
    f = -y**2 + z
    g = x - y**3

    assert groebner([f, g], R, algorithm) == [x - y*z, y**2 - z]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_Buchberger_case14(algorithm):
    R, x,y,z = ring("x,y,z", QQ, grlex)
    f = -y**2 + z
    g = x - y**3

    assert groebner([f, g], R, algorithm) == [-x**2 + z**3, x*y - z**2, y**2 - z, -x + y*z]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_Buchberger_case15(algorithm):
    R, x,y,z = ring("x,y,z", QQ, lex)
    f = y - z**2
    g = x - z**3

    assert groebner([f, g], R, algorithm) == [x - z**3, y - z**2]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_Buchberger_case16(algorithm):
    R, x,y,z = ring("x,y,z", QQ, grlex)
    f = y - z**2
    g = x - z**3

    assert groebner([f, g], R, algorithm) == [-x**2 + y**3, x*z - y**2, -x + y*z, -y + z**2]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_Buchberger_case17(algorithm):
    R, x,y,z = ring("x,y,z", QQ, lex)
    f = 4*x**2*y**2 + 4*x*y + 1
    g = x**2 + y**2 - 1

    assert groebner([f, g], R, algorithm) == [
        x - 4*y**7 + 8*y**5 - 7*y**3 + 3*y,
        y**8 - 2*y**6 + QQ(3,2)*y**4 - QQ(1,2)*y**2 + QQ(1,16),
    ]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_empty(algorithm):
    R, x,y = ring("x,y", QQ, lex)

    assert groebner([], R) == []

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_benchmark_katsura_3(algorithm):
    R, x0,x1,x2 = ring("x:3", ZZ, lex)
    ideal = [x0 + 2*x1 + 2*x2 - 1,
         x0**2 + 2*x1**2 + 2*x2**2 - x0,
         2*x0*x1 + 2*x1*x2 - x1]

    assert groebner(ideal, R, algorithm) == [
        -7 + 7*x0 + 8*x2 + 158*x2**2 - 420*x2**3,
        7*x1 + 3*x2 - 79*x2**2 + 210*x2**3,
        x2 + x2**2 - 40*x2**3 + 84*x2**4,
    ]

    R, x0,x1,x2 = ring("x:3", ZZ, grlex)
    ideal = [ i.set_ring(R) for i in ideal ]

    assert groebner(ideal, R, algorithm) == [
        7*x1 + 3*x2 - 79*x2**2 + 210*x2**3,
        -x1 + x2 - 3*x2**2 + 5*x1**2,
        -x1 - 4*x2 + 10*x1*x2 + 12*x2**2,
        -1 + x0 + 2*x1 + 2*x2,
    ]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_benchmark_katsura_4(algorithm):
    R, x0,x1,x2,x3 = ring("x:4", ZZ, lex)
    ideal = [x0 + 2*x1 + 2*x2 + 2*x3 - 1,
         x0**2 + 2*x1**2 + 2*x2**2 + 2*x3**2 - x0,
         2*x0*x1 + 2*x1*x2 + 2*x2*x3 - x1,
         x1**2 + 2*x0*x2 + 2*x1*x3 - x2]

    assert groebner(ideal, R, algorithm) == [
        5913075*x0 - 159690237696*x3**7 + 31246269696*x3**6 + 27439610544*x3**5 - 6475723368*x3**4 - 838935856*x3**3 + 275119624*x3**2 + 4884038*x3 - 5913075,
        1971025*x1 - 97197721632*x3**7 + 73975630752*x3**6 - 12121915032*x3**5 - 2760941496*x3**4 + 814792828*x3**3 - 1678512*x3**2 - 9158924*x3,
        5913075*x2 + 371438283744*x3**7 - 237550027104*x3**6 + 22645939824*x3**5 + 11520686172*x3**4 - 2024910556*x3**3 - 132524276*x3**2 + 30947828*x3,
        128304*x3**8 - 93312*x3**7 + 15552*x3**6 + 3144*x3**5 -
        1120*x3**4 + 36*x3**3 + 15*x3**2 - x3,
    ]

    R, x0,x1,x2,x3 = ring("x:4", ZZ, grlex)
    ideal = [ i.set_ring(R) for i in ideal ]

    assert groebner(ideal, R, algorithm) == [
        393*x1 - 4662*x2**2 + 4462*x2*x3 - 59*x2 + 224532*x3**4 - 91224*x3**3 - 678*x3**2 + 2046*x3,
        -x1 + 196*x2**3 - 21*x2**2 + 60*x2*x3 - 18*x2 - 168*x3**3 + 83*x3**2 - 9*x3,
        -6*x1 + 1134*x2**2*x3 - 189*x2**2 - 466*x2*x3 + 32*x2 - 630*x3**3 + 57*x3**2 + 51*x3,
        33*x1 + 63*x2**2 + 2268*x2*x3**2 - 188*x2*x3 + 34*x2 + 2520*x3**3 - 849*x3**2 + 3*x3,
        7*x1**2 - x1 - 7*x2**2 - 24*x2*x3 + 3*x2 - 15*x3**2 + 5*x3,
        14*x1*x2 - x1 + 14*x2**2 + 18*x2*x3 - 4*x2 + 6*x3**2 - 2*x3,
        14*x1*x3 - x1 + 7*x2**2 + 32*x2*x3 - 4*x2 + 27*x3**2 - 9*x3,
        x0 + 2*x1 + 2*x2 + 2*x3 - 1,
    ]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_benchmark_czichowski(algorithm):
    R, x,t = ring("x,t", ZZ, lex)
    ideal = [9*x**8 + 36*x**7 - 32*x**6 - 252*x**5 - 78*x**4 + 468*x**3 + 288*x**2 - 108*x + 9,
         (-72 - 72*t)*x**7 + (-256 - 252*t)*x**6 + (192 + 192*t)*x**5 + (1280 + 1260*t)*x**4 + (312 + 312*t)*x**3 + (-404*t)*x**2 + (-576 - 576*t)*x + 96 + 108*t]

    assert groebner(ideal, R, algorithm) == [
        3725588592068034903797967297424801242396746870413359539263038139343329273586196480000*x -
        160420835591776763325581422211936558925462474417709511019228211783493866564923546661604487873*t**7 -
        1406108495478033395547109582678806497509499966197028487131115097902188374051595011248311352864*t**6 -
        5241326875850889518164640374668786338033653548841427557880599579174438246266263602956254030352*t**5 -
        10758917262823299139373269714910672770004760114329943852726887632013485035262879510837043892416*t**4 -
        13119383576444715672578819534846747735372132018341964647712009275306635391456880068261130581248*t**3 -
        9491412317016197146080450036267011389660653495578680036574753839055748080962214787557853941760*t**2 -
        3767520915562795326943800040277726397326609797172964377014046018280260848046603967211258368000*t -
        632314652371226552085897259159210286886724229880266931574701654721512325555116066073245696000,
        610733380717522355121*t**8 +
        6243748742141230639968*t**7 +
        27761407182086143225024*t**6 +
        70066148869420956398592*t**5 +
        109701225644313784229376*t**4 +
        109009005495588442152960*t**3 +
        67072101084384786432000*t**2 +
        23339979742629593088000*t +
        3513592776846090240000,
    ]

    R, x,t = ring("x,t", ZZ, grlex)
    ideal = [ i.set_ring(R) for i in ideal ]

    assert groebner(ideal, R, algorithm) == [
        16996618586000601590732959134095643086442*t**3*x -
        32936701459297092865176560282688198064839*t**3 +
        78592411049800639484139414821529525782364*t**2*x -
        120753953358671750165454009478961405619916*t**2 +
        120988399875140799712152158915653654637280*t*x -
        144576390266626470824138354942076045758736*t +
        60017634054270480831259316163620768960*x**2 +
        61976058033571109604821862786675242894400*x -
        56266268491293858791834120380427754600960,
        576689018321912327136790519059646508441672750656050290242749*t**4 +
        2326673103677477425562248201573604572527893938459296513327336*t**3 +
        110743790416688497407826310048520299245819959064297990236000*t**2*x +
        3308669114229100853338245486174247752683277925010505284338016*t**2 +
        323150205645687941261103426627818874426097912639158572428800*t*x +
        1914335199925152083917206349978534224695445819017286960055680*t +
        861662882561803377986838989464278045397192862768588480000*x**2 +
        235296483281783440197069672204341465480107019878814196672000*x +
        361850798943225141738895123621685122544503614946436727532800,
       -117584925286448670474763406733005510014188341867*t**3 +
        68566565876066068463853874568722190223721653044*t**2*x -
        435970731348366266878180788833437896139920683940*t**2 +
        196297602447033751918195568051376792491869233408*t*x -
        525011527660010557871349062870980202067479780112*t +
        517905853447200553360289634770487684447317120*x**3 +
        569119014870778921949288951688799397569321920*x**2 +
        138877356748142786670127389526667463202210102080*x -
        205109210539096046121625447192779783475018619520,
       -3725142681462373002731339445216700112264527*t**3 +
        583711207282060457652784180668273817487940*t**2*x -
        12381382393074485225164741437227437062814908*t**2 +
        151081054097783125250959636747516827435040*t*x**2 +
        1814103857455163948531448580501928933873280*t*x -
        13353115629395094645843682074271212731433648*t +
        236415091385250007660606958022544983766080*x**2 +
        1390443278862804663728298060085399578417600*x -
        4716885828494075789338754454248931750698880,
    ]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_benchmark_cyclic_4(algorithm):
    R, a,b,c,d = ring("a,b,c,d", ZZ, lex)

    ideal = [a + b + c + d,
         a*b + a*d + b*c + b*d,
         a*b*c + a*b*d + a*c*d + b*c*d,
         a*b*c*d - 1]

    assert groebner(ideal, R, algorithm) == [
        4*a + 3*d**9 - 4*d**5 - 3*d,
        4*b + 4*c - 3*d**9 + 4*d**5 + 7*d,
        4*c**2 + 3*d**10 - 4*d**6 - 3*d**2,
        4*c*d**4 + 4*c - d**9 + 4*d**5 + 5*d, d**12 - d**8 - d**4 + 1
    ]

    R, a,b,c,d = ring("a,b,c,d", ZZ, grlex)
    ideal = [ i.set_ring(R) for i in ideal ]

    assert groebner(ideal, R, algorithm) == [
        3*b*c - c**2 + d**6 - 3*d**2,
        -b + 3*c**2*d**3 - c - d**5 - 4*d,
        -b + 3*c*d**4 + 2*c + 2*d**5 + 2*d,
        c**4 + 2*c**2*d**2 - d**4 - 2,
        c**3*d + c*d**3 + d**4 + 1,
        b*c**2 - c**3 - c**2*d - 2*c*d**2 - d**3,
        b**2 - c**2, b*d + c**2 + c*d + d**2,
        a + b + c + d
    ]

@pytest.mark.parametrize('algorithm', [(_buchberger)])
def test_benchmark_coloring(algorithm):
    V = range(1, 12 + 1)
    E = [(1, 2), (2, 3), (1, 4), (1, 6), (1, 12), (2, 5), (2, 7), (3, 8), (3, 10),
         (4, 11), (4, 9), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11),
         (11, 12), (5, 12), (5, 9), (6, 10), (7, 11), (8, 12), (3, 4)]

    R, V = xring([ "x%d" % v for v in V ], QQ, lex)
    E = [(V[i - 1], V[j - 1]) for i, j in E]

    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = V

    I3 = [x**3 - 1 for x in V]
    Ig = [x**2 + x*y + y**2 for x, y in E]

    ideal = I3 + Ig

    assert groebner(ideal[:-1], R) == [
        x1 + x11 + x12,
        x2 - x11,
        x3 - x12,
        x4 - x12,
        x5 + x11 + x12,
        x6 - x11,
        x7 - x12,
        x8 + x11 + x12,
        x9 - x11,
        x10 + x11 + x12,
        x11**2 + x11*x12 + x12**2,
        x12**3 - 1,
    ]

    assert groebner(ideal, R, algorithm) == [1]

@pytest.mark.parametrize('algorithm', [(_f5b), (_buchberger)])
def test_benchmark_minpoly(algorithm):
    R, x,y,z = ring("x,y,z", QQ, lex)

    F = [x**3 + x + 1, y**2 + y + 1, (x + y) * z - (x**2 + y)]
    G = [x + QQ(155,2067)*z**5 - QQ(355,689)*z**4 + QQ(6062,2067)*z**3 - QQ(3687,689)*z**2 + QQ(6878,2067)*z - QQ(25,53),
         y + QQ(4,53)*z**5 - QQ(91,159)*z**4 + QQ(523,159)*z**3 - QQ(387,53)*z**2 + QQ(1043,159)*z - QQ(308,159),
         z**6 - 7*z**5 + 41*z**4 - 82*z**3 + 89*z**2 - 46*z + 13]

    assert groebner(F, R, algorithm) == G
