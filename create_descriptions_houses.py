'''
The script has been developed to create description for each house
'''
import random
import torch
import inflect
import pickle
from tqdm import tqdm
import time


def num_to_words(num):
    p = inflect.engine()
    return p.number_to_words(num)


def get_theme_desc(i):
    if i['theme'] is not None:
        return ", with " + i['theme'] + " theme"
    return ""


def get_theme(i):
    return i['theme']


def get_material_desc(i):
    material = get_material(i)
    if material is not None and material != "Others":
        return ", and " + material + " material"
    return ""


def get_material(i):
    return i['material']


def get_style_desc(i):
    style = get_style(i)
    if style != "Others":
        if 'style' in style:
            return " with " + style + " "
        else:
            return " with " + style + " style "
    return ""


def get_style(i):
    if i['style'] == "Vintage/Retro":
        return 'Vintage'
    return i['style']


def get_category(i):
    if i['category'] == "Footstool / Sofastool / Bed End Stool / Stool":
        return "stool"
    if i['category'] == "Lounge Chair / Cafe Chair / Office Chair":
        return "chair"
    if i['category'] == "Sideboard / Side Cabinet / Console Table":
        return "Sideboard"
    if i['category'] == "Drawer Chest / Corner cabinet":
        return "Corner cabinet"
    if i['category'] == "Corner/Side Table":
        return "Corner Table"
    if i['category'] == "Bookcase / jewelry Armoire":
        return "Bookcase Armoire"
    return i['category'].lower()


def get_transitional_word():
    words_list = ["Also,", "Moreover,", "Additionally,", "Furthermore,"]
    random_element = random.choice(words_list)
    return random_element


def get_final_word():
    words_list = ["Finally,", "Eventually,", "Ultimately,"]
    random_element = random.choice(words_list)
    return random_element


def get_verb():
    words_list = ["contains", "comprises", "includes"]
    random_element = random.choice(words_list)
    return random_element


def get_description(i, j):
    category = get_category(i[j])
    style = get_style_desc(i[j])
    material = get_material_desc(i[j])
    theme = get_theme_desc(i[j])
    bs = category
    if style != "":
        if "style" not in style:
            bs += style + "style"
        else:
            bs += style
    if material != "":
        bs += material
    if theme != "":
        bs += theme
    return bs.lower()


def get_embeddings(description, model, tokenizer, device):
    tokenized_sentence = description.split('.')
    if tokenized_sentence[-1].replace(" ", "") != '':
        print('tokenized sentence last part: ', tokenized_sentence[-1])
        exit(0)
    obtained_tensor = torch.empty(len(tokenized_sentence) - 1, 768)
    cnt = 0
    with torch.no_grad():
        for idx in range(len(tokenized_sentence) - 1):
            inputs = tokenizer(tokenized_sentence[idx], padding=True, truncation=True, return_tensors='pt')
            inputs = inputs.to(device)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
            sentence_embeddings = torch.mean(embeddings, dim=1)
            obtained_tensor[cnt, :] = sentence_embeddings.cpu()
            cnt += 1
    return obtained_tensor


def create_db(room):
    objects = []
    for idx in range(len(room)):
        category = get_category(room[idx])
        style = get_style(room[idx])
        theme = get_theme(room[idx])
        material = get_material(room[idx])
        temp_dict = {"category": category, "style": style, "theme": theme,
                     "material": material, "number": 1}
        objects.append(temp_dict)

    return objects


def find_uniques(db):
    unique_db = []
    for i in db:
        found = False
        for j in unique_db:
            if not found and j['category'] == i['category'] and j['style'] == i['style'] \
                    and j['theme'] == i['theme'] and j['material'] == i['material']:
                j['number'] += 1
                found = True
        if not found:
            unique_db.append(i)

    return unique_db


def get_type_room(type):
    if type == 'LivingDiningRoom':
        return 'living dining room'
    elif type == 'LivingRoom':
        return 'living room'
    elif type == 'StorageRoom':
        return 'storage room'
    elif type == 'Bathroom':
        return 'bathroom'
    elif type == 'Aisle':
        return 'aisle'
    elif type == 'KidsRoom':
        return 'kids room'
    elif type == 'Kitchen':
        return 'kitchen'
    elif type == 'SecondBathroom':
        return 'second bathroom'
    elif type == 'MasterBathroom':
        return 'master bathroom'
    elif type == 'Lounge':
        return 'lounge'
    elif type == 'Library':
        return 'library'
    elif type == 'Balcony':
        return 'balcony'
    elif type == 'OtherSpace':
        return 'other space'
    elif type == 'ElderlyRoom':
        return 'elderly room'
    elif type == 'LaundryRoom':
        return 'laundry room'
    elif type == 'NannyRoom':
        return 'nanny room'
    elif type == 'CloakRoom':
        return 'cloak room'
    elif type == 'EquipmentRooms':
        return 'equipment room'
    elif type == 'MasterBedroom':
        return 'master bedroom'
    elif type in ['SecondBedroom', 'Bedroom']:
        return 'bedroom'
    elif type == 'OtherRoom':
        return 'other room'
    else:
        return type.lower()


def create_rooms_descs(rooms_data):
    tmp_dict = {}
    for r in rooms_data:
        get_type = get_type_room(r)
        if get_type in tmp_dict.keys():
            tmp_dict[get_type_room(r)] += 1
        else:
            tmp_dict[get_type_room(r)] = 1
    tmp_sent = ''
    for idx, tmp in enumerate(tmp_dict):
        correct_format = tmp
        if tmp_dict[tmp] > 1:
            if tmp[-1] == 'y':
                correct_format = tmp[:-1] + 'ies'
            else:
                correct_format = tmp + 's'
        if idx < (len(tmp_dict) - 1) or len(tmp_dict) == 1:
            tmp_sent += num_to_words(tmp_dict[tmp]) + ' ' + correct_format + ', '
        else:
            tmp_sent += 'and ' + num_to_words(tmp_dict[tmp]) + ' ' + correct_format + '.'
    return tmp_sent


def process_rooms(rooms_data):
    living_rooms = []
    bedrooms = []

    bedroom_types = ['Bedroom', 'MasterBedroom', 'SecondBedroom']
    living_room_types = ['LivingDiningRoom', 'LivingRoom']
    for r in rooms_data:
        if r['type'] in bedroom_types:
            bedrooms.append(r)
        elif r['type'] in living_room_types:
            living_rooms.append(r)
    return living_rooms, bedrooms


def add_room_objects_descriptions(objects_data):
    number_of_sent = 0
    if len(objects_data) == 0:
        return 'is completely empty.'
    base_sent = ''
    for ii, jj in enumerate(objects_data):
        if jj['number'] == 1:
            if number_of_sent == 0:
                base_sent += get_verb() + " one " + jj['category'] + get_style_desc(jj) \
                                    + get_theme_desc(jj) + get_material_desc(jj) + "."
            else:
                base_sent += " " + get_transitional_word() + " it " + get_verb() + " one " + jj['category'] \
                                    + get_style_desc(jj) + get_theme_desc(jj) + get_material_desc(jj) + "."
        else:
            word_of_num = num_to_words(jj['number'])
            if number_of_sent == 0:
                base_sent += get_verb() + ' ' + word_of_num + " " + jj['category'] + get_style_desc(jj) \
                                    + get_theme_desc(jj) + get_material_desc(jj) + "."
            else:
                base_sent += " " + get_transitional_word() + " it " + get_verb() + " " + word_of_num + " " \
                                    + jj['category'] + get_style_desc(jj) + get_theme_desc(jj) + get_material_desc(
                    jj) + "."
        number_of_sent += 1
    return base_sent


def get_rank_exp(no_of_bedroom):
    if no_of_bedroom == 1:
        return 'first'
    elif no_of_bedroom == 2:
        return 'second'
    elif no_of_bedroom == 3:
        return 'third'
    elif no_of_bedroom == 4:
        return 'fourth'
    elif no_of_bedroom == 5:
        return 'fifth'
    elif no_of_bedroom == 6:
        return 'sixth'
    elif no_of_bedroom == 7:
        return 'seventh'
    else:
        return 'last'


def create_descriptions(houses_data):

    empty_living_rooms = []
    empty_bedrooms = []

    for house_idx, house in tqdm(enumerate(houses_data), total=len(houses_data)):
        basic_sentence = 'This house contains ' + num_to_words(len(house['entire_rooms'])) + ' rooms. '
        basic_sentence += 'More specifically, it has ' + create_rooms_descs(house['entire_rooms']) + ' '
        living_rooms, bedrooms = process_rooms(house['rooms'])
        no_living_room = len(living_rooms)
        no_bedrooms = len(bedrooms)

        living_rooms_sent = ''
        if no_living_room == 1:
            living_rooms_sent += 'Regarding the ' + get_type_room(living_rooms[0]['type']) + ', it '
            db_objects_living_room = create_db(living_rooms[0]['objects'])
            processed_db_objects_living_room = find_uniques(db_objects_living_room)
            living_rooms_sent += add_room_objects_descriptions(objects_data=processed_db_objects_living_room)
        if no_living_room > 1:
            living_rooms_sent += 'Regarding the living rooms, '
            for idx_room in range(len(living_rooms)):
                if idx_room == 0:
                    if get_type_room(living_rooms[idx_room]['type']) == 'living room':
                        living_rooms_sent += 'the ' + get_rank_exp(idx_room + 1) + ' one '
                    else:
                        living_rooms_sent += 'the ' + get_rank_exp(idx_room + 1) + ' one, which is a ' \
                                             + get_type_room(living_rooms[0]['type']) + ' '
                else:
                    if get_type_room(living_rooms[idx_room]['type']) == 'living room':
                        living_rooms_sent += ' The ' + get_rank_exp(idx_room + 1) + ' living room '
                    else:
                        living_rooms_sent += ' The ' + get_rank_exp(idx_room + 1) + ' living room, which is a ' \
                                             + get_type_room(living_rooms[idx_room]['type']) + ' '
                db_objects_living_room = create_db(living_rooms[idx_room]['objects'])
                if len(db_objects_living_room) == 0:
                    empty_living_rooms.append(house_idx)
                    # exit(0)
                processed_db_objects_living_room = find_uniques(db_objects_living_room)
                living_rooms_sent += add_room_objects_descriptions(objects_data=processed_db_objects_living_room)

        bedrooms_sent = ''
        if no_bedrooms == 1:
            bedrooms_sent += ' Regarding the ' + get_type_room(bedrooms[0]['type']) + ', it '
            db_objects_bedroom = create_db(bedrooms[0]['objects'])
            processed_db_objects_bedroom = find_uniques(db_objects_bedroom)
            bedrooms_sent += add_room_objects_descriptions(objects_data=processed_db_objects_bedroom)
        if no_bedrooms > 1:
            bedrooms_sent += ' Regarding the bedrooms, '
            for idx_room in range(len(bedrooms)):
                if idx_room == 0:
                    if get_type_room(bedrooms[idx_room]['type']) == 'bedroom':
                        bedrooms_sent += 'the ' + get_rank_exp(idx_room + 1) + ' one '
                    else:
                        bedrooms_sent += 'the ' + get_rank_exp(idx_room + 1) + ' one, which is a ' \
                                         + get_type_room(bedrooms[idx_room]['type']) + ' '
                else:
                    if get_type_room(bedrooms[idx_room]['type']) == 'bedroom':
                        bedrooms_sent += ' The ' + get_rank_exp(idx_room + 1) + ' bedroom '
                    else:
                        bedrooms_sent += ' The ' + get_rank_exp(idx_room + 1) + ' bedroom, which is a ' \
                                             + get_type_room(bedrooms[idx_room]['type']) + ' '
                db_objects_bedroom = create_db(bedrooms[idx_room]['objects'])
                if len(db_objects_bedroom) == 0:
                    empty_bedrooms.append(house_idx)
                    # exit(0)
                processed_db_objects_bedroom = find_uniques(db_objects_bedroom)
                bedrooms_sent += add_room_objects_descriptions(objects_data=processed_db_objects_bedroom)
        entire_desc = basic_sentence + living_rooms_sent + bedrooms_sent

        with open('descriptions/' + house['json_file'] + '.txt', 'w') as f:
            f.write(entire_desc)

    # print('no empty living rooms', len(empty_living_rooms))
    # print('no empty bedrooms', len(empty_bedrooms))
    # with open('./empty_rooms/living_room.pkl', 'wb') as f:
    #     pickle.dump(empty_living_rooms, f)
    # with open('./empty_rooms/bedroom.pkl', 'wb') as f:
    #     pickle.dump(empty_bedrooms, f)


if __name__ == '__main__':
    start_time = time.time()
    houses_data = pickle.load(open('./houses_data/houses_data.pkl', 'rb'))
    create_descriptions(houses_data)
    print('Descriptions Creation Finished')
