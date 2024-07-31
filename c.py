import cv2
import runner
for res, img in runner.classifier(videoCaptureDeviceId):
    if (next_frame > now()):
        time.sleep((next_frame - now()) / 1000)
        #print('classification runner response', res)
    if "classification" in res["result"].keys():
        print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
        for label in labels:
            score = res['result']['classification'][label]
            print('%s: %.2f\t' % (label, score), end='')
            print('', flush=True)
    elif "bounding_boxes" in res["result"].keys():
        print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
        prices = {"cadbury_DM" : 1.1, "indomie_goreng" : 0.4, "kitkat" : 0.6, "kitkat_gold" : 0.8, "mentos" : 0.7, "milo_nuggets" : 1.0, "pocky_chocolate" : 1.2, "toblerone" : 2.0};   # set item price
        total = 0
        for bb in res["result"]["bounding_boxes"]:
            print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
            img = cv2.rectangle(img, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)
            total += prices[bb['label']]    #set total price
            print("Writing to display") # write to 16x2 LCD
            display.lcd_display_string("Items: " + str(len(res["result"]["bounding_boxes"])), 1) # show total bounding boxes as items  
            display.lcd_display_string("Total: $" + "{:.2f}".format(total), 2) # show total price
            if (show_camera):
                cv2.imshow('edgeimpulse', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) == ord('q'):
                    break