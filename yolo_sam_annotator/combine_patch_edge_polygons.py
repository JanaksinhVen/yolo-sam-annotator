from shapely.geometry import Polygon
import json
import cv2
import numpy as np
# finally worked
def polygons_to_mask(polygons, width, height):
    mask = np.zeros((width, height), dtype=np.uint8)
    for polygon in polygons:
        poly = np.array(polygon).reshape((1, -1, 2)).astype(np.int32)
        cv2.fillPoly(mask, poly, 1)
    return mask

def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [contour.flatten().tolist() for contour in contours]
    return polygons

def combine_polygons_left_right(polygon_1,polygon_2,p_width, p_height):
    for i in range(len(polygon_2)):
        polygon_2[i] = [polygon_2[i][0]+p_width, polygon_2[i][1]]
  
    mask1 = polygons_to_mask([polygon_1], p_height*2,p_width*2)
    mask2 = polygons_to_mask([polygon_2], p_height*2,p_width*2)
    combined_mask = np.maximum(mask1, mask2)
    # plt.imshow(combined_mask)
    combined_polygons = mask_to_polygons(combined_mask)
    if len(combined_polygons)==1:
        print('merged')
        print(f'The #polygon= {len(combined_polygons)}')
    else:
        print(f'The #polygon= {len(combined_polygons)}')
    return [[combined_polygons[0][i], combined_polygons[0][i + 1]] for i in range(0, len(combined_polygons[0]), 2)]


def combine_polygons_top_bottom(polygon_1,polygon_2,p_width, p_height):

    for i in range(len(polygon_2)):
        polygon_2[i] = [polygon_2[i][0], polygon_2[i][1]+p_height]

    mask1 = polygons_to_mask([polygon_1], p_height*2, p_width*2)
    mask2 = polygons_to_mask([polygon_2], p_height*2, p_width*2)
    combined_mask = np.maximum(mask1, mask2)
    combined_polygons = mask_to_polygons(combined_mask)
    if len(combined_polygons)==1:
        print('merged')
        print(f'The #polygon= {len(combined_polygons)}')
    else:
        print(f'The #polygon= {len(combined_polygons)}')
    return [[combined_polygons[0][i], combined_polygons[0][i + 1]] for i in range(0, len(combined_polygons[0]), 2)]
    
def combine_polygons_rb_lt_corner(polygon_1,polygon_2,p_width, p_height):
    for i in range(len(polygon_2)):
        polygon_2[i] = [polygon_2[i][0]+p_width, polygon_2[i][1]+p_height]
    mask1 = polygons_to_mask([polygon_1], p_height*2,p_width*2)
    mask2 = polygons_to_mask([polygon_2], p_height*2,p_width*2)
    combined_mask = np.maximum(mask1, mask2)
    # plt.imshow(combined_mask)
    combined_polygons = mask_to_polygons(combined_mask)
    if len(combined_polygons)==1:
        print('merged')
        print(f'The #polygon= {len(combined_polygons)}')
    else:
        print(f'The #polygon= {len(combined_polygons)}')
    return [[combined_polygons[0][i], combined_polygons[0][i + 1]] for i in range(0, len(combined_polygons[0]), 2)]
# match_thresold=50,threshold=15)
def match_left_right_poly(polygon_1,polygon_2,p_width, p_height,m_thresold,threshold=15):
    list_i=[]
    s=0
    d=0
    for i,point in enumerate(polygon_1):
        if point[0]>=(p_width-threshold):
            d+=1
            s+= point[1]
    avg_1 = s/d
    s=0
    d=0
    for i,point in enumerate(polygon_2):
        if point[0]<=threshold:
            d+=1
            s+= point[1]
    avg_2 = s/d

    return abs(avg_1-avg_2) < m_thresold

def match_top_bottom_poly(polygon_1,polygon_2,p_width, p_height,m_thresold,threshold=15):
    s=0
    d=0
    for i,point in enumerate(polygon_1):
        # print('in')
        if point[1]>=(p_height-threshold):
            d+=1
            s+= point[0]
    
    avg_1 = s/d
    s=0
    d=0
    for i,point in enumerate(polygon_2):
        if point[1]<=threshold:
            d+=1
            s+= point[0]
    avg_2 = s/d

    return abs(avg_1-avg_2) < m_thresold


def load_json_file(input_filepath):
    with open(input_filepath, 'r') as file:
        data = json.load(file)
    return data
def save_json_file(output_filepath, data):
    with open(output_filepath, 'w') as file:
        json.dump(data, file, indent=1)

def is_polygon_on_boundary(polygon, image_width, image_height, side, threshold=15):
    # Create the polygon object
    poly = Polygon(polygon)
    # Get the coordinates of the polygon's vertices
    min_x, min_y, max_x, max_y = poly.bounds

    # Define conditions for each boundary
    if side == 'left':
        return min_x < threshold 
    elif side == 'right':
        return max_x > (image_width - threshold) and max_x < image_width
    elif side == 'top':
        return min_y < threshold
    elif side == 'bottom':
        return max_y > (image_height - threshold) and max_y < image_height
    elif side == 'r_b_corner':
        return max_x > image_width and max_y > image_height
    elif side == 'l_t_corner':
        return min_x < threshold and min_y < threshold
    else:
        raise ValueError("Side must be one of ['left', 'right', 'top', 'bottom']")


def func(i,all_polygons):
    if all_polygons[f'image{i}']['polygons']==[]:
        # print(f'skip-{i}')
        pass
    else:
        print('1',i)

        polygons_i_1= all_polygons[f'image{i}']['polygons']
        polygons_label_1 = all_polygons[f'image{i}']['labels']
   
        if (i-9)%(image_size[0]/patch_size[1]) != 0:
            print('2',i)
            polygons_i_2= all_polygons[f'image{i+1}']['polygons']
            polygons_label_2 = all_polygons[f'image{i+1}']['labels']
            side = 'right'
            boundary_indexes1 = []
            for j, polygon in enumerate(polygons_i_1):
                
                if is_polygon_on_boundary(polygon, image_width, image_height, side, threshold):
                    print('index_right:',i,j)
                    boundary_indexes1.append(j)
            if boundary_indexes1!=[]:
                

                side_2 = 'left'
                boundary_indexes2 = []
                for k, polygon in enumerate(polygons_i_2):
                    
                    if is_polygon_on_boundary(polygon, image_width, image_height, side_2, threshold):
                        print('index_left:',i,k)
                        boundary_indexes2.append(k)

                boundary_indexes1.sort(reverse=True)
                boundary_indexes2.sort(reverse=True)
                del_poly_index_1 = []
                del_poly_index_2 = []
                com_polygons = []
                com_polygons_label = []
                while(boundary_indexes1!=[] and boundary_indexes2!=[]):
                    print('in while',boundary_indexes1,boundary_indexes2)
                    for a in boundary_indexes1:
                        for b in boundary_indexes2:
                            if polygons_label_1[a]==polygons_label_2[b]:
                                print(a,b)
                                if match_left_right_poly(polygons_i_1[a],polygons_i_2[b],patch_size[0],patch_size[1],match_thresold,threshold):
                                    print(f'image{i} right matched ')
                                    c_polygon = combine_polygons_left_right(polygons_i_1[a],polygons_i_2[b],patch_size[1],patch_size[0])
                                    
                                    l_temp=polygons_label_1[a]
                                    del_poly_index_1.append(a)
                                    del_poly_index_2.append(b)
                                    com_polygons.append(c_polygon)
                                    com_polygons_label.append(l_temp)
                               
                                    boundary_indexes2. remove(b)
                                    boundary_indexes1. remove(a)
                                    break
                        if a in boundary_indexes1:
                            boundary_indexes1. remove(a)
                del_poly_index_1.sort(reverse=True)
                del_poly_index_2.sort(reverse=True)
                print('del_poly_index_',del_poly_index_1,del_poly_index_2,com_polygons_label)
                for a in del_poly_index_1:
                    del polygons_i_1[a]
                    del polygons_label_1[a]
                for b in del_poly_index_2:
                    del polygons_i_2[b]
                    del polygons_label_2[b]
                for l,c in enumerate(com_polygons):
                    polygons_i_1.append(c)
                    polygons_label_1.append(com_polygons_label[l])
                    
            all_polygons[f'image{i+1}']['polygons']=polygons_i_2
            all_polygons[f'image{i+1}']['labels']=polygons_label_2

        if i<(n_patch+10-(image_size[0]/patch_size[0])):
            print('3',i)
            polygons_i_2= all_polygons[f'image{i+6}']['polygons']
            polygons_label_2 = all_polygons[f'image{i+6}']['labels']
            side = 'bottom'
            boundary_indexes1 = []
            for j, polygon in enumerate(polygons_i_1):
                
                if is_polygon_on_boundary(polygon, image_width, image_height, side, threshold):
                    print('index_bottom:',i,j)
                    boundary_indexes1.append(j)
            if boundary_indexes1!=[]:
                

                side_2 = 'top'
                boundary_indexes2 = []
                for k, polygon in enumerate(polygons_i_2):
                    
                    if is_polygon_on_boundary(polygon, image_width, image_height, side_2, threshold):
                        print('index_top:',i,k)
                        boundary_indexes2.append(k)
                boundary_indexes1.sort(reverse=True)
                boundary_indexes2.sort(reverse=True)
         
                del_poly_index_1 = []
                del_poly_index_2 = []
                com_polygons = []
                com_polygons_label = []
                while(boundary_indexes1!=[] and boundary_indexes2!=[]):
                    print('in while',boundary_indexes1,boundary_indexes2)
                  
                    for a in boundary_indexes1:
                        # f+=1
                        for b in boundary_indexes2:
                            # if f==1:
                                # continue
                            print('out',a,b)
                            
                            if polygons_label_1[a]==polygons_label_2[b]:
                                print('in',a,b)
                                if match_top_bottom_poly(polygons_i_1[a],polygons_i_2[b],patch_size[1],patch_size[0],match_thresold,threshold):
                                    # f=1
                                    print(f'image{i} bottom matched ')
                                    c_polygon = combine_polygons_top_bottom(polygons_i_1[a],polygons_i_2[b],patch_size[1],patch_size[0])
                                    l_temp=polygons_label_1[a]

                                    del_poly_index_1.append(a)
                                    del_poly_index_2.append(b)
                                    com_polygons.append(c_polygon)
                                    com_polygons_label.append(l_temp)
                               
                                    boundary_indexes2. remove(b)
                                    boundary_indexes1. remove(a)
                               
                                    break
                        if a in boundary_indexes1:
                            boundary_indexes1. remove(a)
                del_poly_index_1.sort(reverse=True)
                del_poly_index_2.sort(reverse=True)
                print('del_poly_index_',del_poly_index_1,del_poly_index_2,com_polygons_label)
                for a in del_poly_index_1:
                    del polygons_i_1[a]
                    del polygons_label_1[a]
                for b in del_poly_index_2:
                    del polygons_i_2[b]
                    del polygons_label_2[b]
                for l,c in enumerate(com_polygons):
                    polygons_i_1.append(c)
                    polygons_label_1.append(com_polygons_label[l])


                    
            all_polygons[f'image{i+6}']['polygons']=polygons_i_2
            all_polygons[f'image{i+6}']['labels']=polygons_label_2
# corner
        if i<(n_patch+10-(image_size[0]/patch_size[0])) and (i-9)%(image_size[0]/patch_size[1]) != 0:
            print('4',i)
            polygons_i_2= all_polygons[f'image{i+6+1}']['polygons']
            polygons_label_2 = all_polygons[f'image{i+6+1}']['labels']
            side = 'r_b_corner'
            boundary_indexes1 = []
            for j, polygon in enumerate(polygons_i_1):
                
                if is_polygon_on_boundary(polygon, image_width, image_height, side, threshold):
                    print('index_r_b_corner',i,j)
                    boundary_indexes1.append(j)
            if boundary_indexes1!=[]:
                

                side_2 = 'l_t_corner'
                boundary_indexes2 = []
                for k, polygon in enumerate(polygons_i_2):
                    
                    if is_polygon_on_boundary(polygon, image_width, image_height, side_2, threshold):
                        print('index_l_t_corner:',i,k)
                        boundary_indexes2.append(k)
                boundary_indexes1.sort(reverse=True)
                boundary_indexes2.sort(reverse=True)
         
                d=0
                for a in boundary_indexes1:
                    # f=0
                    for b in boundary_indexes2:
                        # if f==1:
                            # continue
                        if polygons_label_1[a]==polygons_label_2[b]:
                            print('label matched for indexes:', a,b)
                        
                            c_polygon = combine_polygons_rb_lt_corner(polygons_i_1[a],polygons_i_2[b],patch_size[1],patch_size[0])
                            l_temp=polygons_label_1[a]
                            del polygons_i_1[a]
                            del polygons_i_2[b]
                            del polygons_label_1[a]
                            del polygons_label_2[b]
                            boundary_indexes2. remove(b)
                            boundary_indexes1. remove(a)
                            polygons_i_1.append(c_polygon)
                            polygons_label_1.append(l_temp)
                            d=1
                            break
                    if d==1:
                        break
                    
            all_polygons[f'image{i+6+1}']['polygons']=polygons_i_2
            all_polygons[f'image{i+6+1}']['labels']=polygons_label_2

        all_polygons[f'image{i}']['polygons']=polygons_i_1
        all_polygons[f'image{i}']['labels']=polygons_label_1
    return all_polygons
                # store and delete

if __name__=='__main__':

    match_thresold=50
    threshold=20
    patch_size=(1000,1000)
    image_size = (6000,8000)
    image_width,image_height = patch_size[1],patch_size[0]
    # img_name = '05_02_01'
    # img_name = '05_01_08'


    n_patch = int((image_size[0]*image_size[1])/(patch_size[0]*patch_size[1]))
    all_polygons = load_json_file(f'polygons.json')

    for i in range(10,10+n_patch):
        all_polygons = func(i,all_polygons)
            
    save_json_file(f'polygons_combined.json', all_polygons)
    print(f'combined_polygons done and saved in polygons_combined.json')




