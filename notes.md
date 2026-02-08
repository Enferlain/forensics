original test images in test folder:

all images in the test folder are noisy, blurry, and have noisy/blurry/smudged details and texture, sometimes on both the subject and the background, and sometimes mostly the background. 

noisy but less blurry and smudgy 000-00-shiro_black-4.966.png 010-00-shiro_black-5.057.png 011-00-shiro_black-5.882.png 116-00-shiro_black-7.595.png 159-00-shiro_black-6.230.png 199-00-shiro_black-5.248.png 203-00-shiro_black-6.077.png 245-00-shiro_black-5.122.png 310-00-shiro_black-6.733.png

noisy and very smudgy and blurry - 242-00-shiro_black-5.456.png 271-00-shiro_black-5.009.png 294-00-shiro_black-4.710.png

038-04-noob10b-6.081.png is 1536x2048 and noisy overall (invisible mostly), but not blurry on details, and there are no visible texture artifacts

00166-4154668875.png is clean, should score the highest for texture and subject quality (other than 0.38 but that one is 2x pixels)

---

rule for new test images (manual labels) in test_new

clean:

Subject edges look crisp at 100% zoom.
No visible smudging/texture mush in key areas (face/hair/clothing edges).
Noise/grain is minimal or only in background.
You would not hesitate to call it “good quality.”

ok:

Subject is mostly clear, but one noticeable issue: light blur, mild smudge, or mild noise on subject.
Edges are soft but still acceptable.
Would pass as “usable,” not “clean.”

bad:

Subject has obvious blur/smudge/noise.
Fine texture is lost (hair strands, fabric detail), or edges look washed out.
You would call it “poor quality.”
