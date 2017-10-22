import tensorflow as tf

with tf.name_scope('input') as input_scope:
    virtues = tf.placeholder(dtype=tf.int32, shape=[2, 5], name="virtues")
    relations = tf.placeholder(dtype=tf.int32, shape=[2, 5], name="relations")
    members = tf.placeholder(dtype=tf.int32, shape=[5, 6, 4], name="members")
    sacrifice = tf.placeholder(dtype=tf.int32, shape=[2], name="sacrifice")



def idea(map_size=2):
    # Value : [HOUSE][0]
    # Threshold : [HOUSE][1]
    virtue_input = tf.placeholder(dtype=tf.int32, shape=[5, 2], name="virtue_input")
    house_input = tf.placeholder(dtype=tf.int32, shape=[5, 2], name="house_input")

    # Alive : [HOUSE][FAMILY][0]
    # Gender : [HOUSE][FAMILY][1]
    # Virtue : [HOUSE][FAMILY][2]
    # Vice : [HOUSE][FAMILY][3]
    cult_input = tf.placeholder(dtype=tf.int32, shape=[5, 6, 4], name="cult_input")

    # Gender : [0]
    # Vice : [1]
    sacrifice_input = tf.placeholder(dtype=tf.int32, shape=2, name="sacrifice_input")

    # Vice Virtue [0]
    # Comparison [1]
    # OUT : Similarity
    virtue_vice_similarity_table = tf.get_variable("virtue_vice_similarity_table", [map_size, map_size],
                                                   initializer=tf.ones_initializer)
    # Vice Virtue [0]
    # OUT : Impact
    virtue_vice_impact_table = tf.get_variable("virtue_vice_impact_table", [map_size, 5],
                                               initializer=tf.zeros_initializer)

    virtue_threshold_weight_table = tf.get_variable("virtue_threshold_weight_table", [5, 3],
                                                    initializer=tf.ones_initializer)

    # Returns a relation between the input virtue/vice, and the comparision virtue/vice This comparison is how
    # similar the two are, IE word association; If i say ignorance it thinks of dull, knows too much,
    # etc MY EXPECTATIONS are that
    # 0 represents no similarity, - represents dissimilarity, and + represents dissimilarity
    def virtue_vice_similarity(vvin, comp, name=None):
        with tf.name_scope(name, "virtue_vice_similarity", [vvin, comp, virtue_vice_similarity_table]):
            # pairs = tf.map_fn(lambda x: (x[0], x[1]), (vvin, comp), dtype=(tf.int32, tf.int32), name="pairs")
            # print(virtue_vice_similarity_table)
            # print(virtue_vice_similarity_table[pairs[0][0][0]])
            # print(virtue_vice_similarity_table[pairs[0][0][0]][pairs[0][0][1]])

            # Todo, expand or find alternative,
            # map_fn doesn't work because of rank issues
            h_list = []
            for h_i in range(5):
                c_list = []
                for c_i in range(6):
                    c_list.append(virtue_vice_similarity_table[vvin[h_i, c_i], comp[h_i, c_i]])
                h_list.append(tf.stack(c_list, name="house"))
            # return
            results = tf.stack(h_list, name="result")
            return results

    def virtue_vice_impact(vvin, name=None):
        with tf.name_scope(name, "virtue_vice_impact", [vvin, virtue_vice_impact_table]):
            # pairs = tf.map_fn(lambda x: (x[0], x[1]), (vvin, comp), dtype=(tf.int32, tf.int32), name="pairs")
            # print(virtue_vice_similarity_table)
            # print(virtue_vice_similarity_table[pairs[0][0][0]])
            # print(virtue_vice_similarity_table[pairs[0][0][0]][pairs[0][0][1]])

            # Todo, expand or find alternative,
            # map_fn doesn't work because of rank issues
            h_list = []
            for h_i in range(5):
                c_list = []
                for c_i in range(6):
                    c_list.append(virtue_vice_impact_table[vvin[h_i, c_i]])
                h_list.append(tf.stack(c_list, name="house"))
            # return
            results = tf.stack(h_list, name="result")
            return results

    # Returns how eligible the cult is given the sacrificial information
    def sacrifice_eligibility(name=None):
        with tf.name_scope(name, "sacrifice_eligibility", [cult_input, sacrifice_input]):
            cult_gender = tf.identity(cult_input[:, :, 1], name="cult_gender")
            cult_vice = tf.identity(cult_input[:, :, 3], name="cult_vice")
            sacrifice_gender = tf.fill(tf.shape(cult_gender), sacrifice_input[0], name="sacrifice_gender")
            sacrifice_vice = tf.fill(tf.shape(cult_vice), sacrifice_input[1], name="sacrifice_vice")
            gender_match = tf.cast(tf.equal(cult_gender, sacrifice_gender), dtype=tf.float32,
                                   name="gender_match_weight")

            vice_match = virtue_vice_similarity(cult_vice, sacrifice_vice, name="vice_match_weight")
            return tf.multiply(gender_match, vice_match, name="eligibility")

    # Returns how eligible the cult is given how alive they are, (hint; if they aren't alive, they aren't eligible)
    def alive_eligibility(name=None):
        with tf.name_scope(name, "sacrifice_eligibility", [cult_input]):
            return tf.cast(cult_input[:, :, 0], dtype=tf.float32, name="eligibility")

    # Returns an eligibility based on how
    def virtue_threshold_eligibility(name=None):
        # Returns the state each threshold is in, 0L, 1E, 2R
        def virtue_threshold_state():
            with tf.name_scope(name, "virtue_threshold_state", [virtue_input]):
                vless = tf.less(virtue_input[:, 0], virtue_input[:, 1])
                vgreat = tf.greater(virtue_input[:, 0], virtue_input[:, 1])
                # vequal = tf.equal(virtue_input[:,0], virtue_input[:,1])
                # v = tf.map_fn(lambda x: (x[0], x[1], x[2]), (vless, vequal, vgreat))
                rless = tf.fill([5], 0)
                requal = tf.fill([5], 1)
                rgreat = tf.fill([5], 2)

                return tf.where(vless, rless, tf.where(vgreat, rgreat, requal))

        def virtue_threshold_weights():
            with tf.name_scope(name, "virtue_threshold_weights", [virtue_threshold_weight_table]):
                # Todo, expand or find alternative,
                # map_fn doesn't work because of rank issues
                v_list = []
                vts = virtue_threshold_state()
                for v_i in range(5):
                    v_list.append(virtue_threshold_weight_table[v_i, vts[v_i]])
                # return
                results = tf.stack(v_list, name="result")
                return results

        with tf.name_scope(name, "virtue_eligibility", [cult_input]):
            cult_virtue = tf.identity(cult_input[:, :, 2], name="cult_virtue")
            cult_virtue_impact = virtue_vice_impact(cult_virtue, name="cult_impact")
            virtue_weights = virtue_threshold_weights()
            result = cult_virtue_impact * virtue_weights
            return tf.reduce_sum(result, axis=2, name="result")

    # Returns how eligible the cult is
    def calculate_eligibility(name=None):
        with tf.name_scope(name, "calculate_eligibility", [cult_input, sacrifice_input]):
            alive = alive_eligibility("alive")
            # Literally, the most important; you cannot be eligible if you're dead
            sacrifice = sacrifice_eligibility(name="sacrifice")
            virtue_threshold = virtue_threshold_eligibility("virtue_threshold")
            print(alive)
            print(sacrifice)
            return alive * (sacrifice + virtue_threshold)

    with tf.Session() as sess:
        virtue_values = [[30, 20], [40, 40], [50, 60], [60, 80], [70, 100]]
        house_values = [[30, 20], [40, 40], [50, 60], [60, 80], [70, 100]]
        god_values = [0, 0]
        court_values = [
            [
                [int(True), int(True), "???", "???"],
                [int(True), int(False), "???", "???"],
                [int(True), int(True), "???", "???"],
                [int(True), int(True), "???", "???"],
                [int(True), int(False), "???", "???"],
                [int(True), int(False), "???", "???"],
            ],
            [
                [int(True), int(True), "???", "???"],
                [int(True), int(False), "???", "???"],
                [int(True), int(True), "???", "???"],
                [int(True), int(True), "???", "???"],
                [int(True), int(False), "???", "???"],
                [int(True), int(False), "???", "???"],
            ],
            [
                [int(True), int(True), "???", "???"],
                [int(True), int(False), "???", "???"],
                [int(True), int(True), "???", "???"],
                [int(True), int(True), "???", "???"],
                [int(True), int(False), "???", "???"],
                [int(True), int(False), "???", "???"],
            ],
            [
                [int(True), int(True), "???", "???"],
                [int(True), int(False), "???", "???"],
                [int(True), int(True), "???", "???"],
                [int(True), int(True), "???", "???"],
                [int(True), int(False), "???", "???"],
                [int(True), int(False), "???", "???"],
            ],
            [
                [int(True), int(True), "???", "???"],
                [int(True), int(False), "???", "???"],
                [int(True), int(True), "???", "???"],
                [int(True), int(True), "???", "???"],
                [int(True), int(False), "???", "???"],
                [int(True), int(False), "???", "???"],
            ]
        ]

        # Fix
        def quick_lookup(q_dict, q_data):
            try:
                return q_dict[q_data]
            except:
                q_dict[q_data] = len(q_dict)
                return q_dict[q_data]

        lookup = {}
        for h, house in enumerate(court_values):
            for m, member in enumerate(house):
                for d, data in enumerate(member):
                    if d == 0 or d == 1:
                        court_values[h][m][d] = int(data)
                    else:
                        court_values[h][m][d] = quick_lookup(lookup, data)
        print(court_values)
        writer = tf.summary.FileWriter("graph", sess.graph)
        sess.run(tf.initialize_all_variables())
        r = sess.run(
            [calculate_eligibility()],
            feed_dict={
                virtue_input: virtue_values,
                house_input: house_values,
                cult_input: court_values,
                sacrifice_input: god_values,
            }
        )
        writer.close()
        print(r)
