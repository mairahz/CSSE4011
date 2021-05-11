/* main.c - Application main entry point */

/*
 * Copyright (c) 2015-2016 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/types.h>
#include <stddef.h>
#include <sys/printk.h>
#include <sys/util.h>
#include <bluetooth/bluetooth.h>
#include <bluetooth/hci.h>

static uint8_t mfg_data[29] = {0x00};

static const struct bt_data ad[] = {
	{
		.type = 0xFF,
		.data_len = 29,
		.data = (const uint8_t *) mfg_data
	}
};

static void scan_cb(const bt_addr_le_t *addr, int8_t rssi, uint8_t adv_type, struct net_buf_simple *buf) {

	mfg_data[0] = 0x63; 
	mfg_data[1] = 0x73; 
	mfg_data[2] = 0x00; 
	mfg_data[3] = 0x09; 

	if ((buf->data[6] == 0x63) && (buf->data[7] == 0x73) && (buf->data[8] == 0x00) && (buf->data[9] == 0x01)) {
		mfg_data[4] = rssi;
		mfg_data[5] = buf->data[10];
	}
	if ((buf->data[6] == 0x63) && (buf->data[7] == 0x73) && (buf->data[8] == 0x00) && (buf->data[9] == 0x02)) {
		mfg_data[6] = rssi;
		mfg_data[7] = buf->data[10];
	}
	if ((buf->data[6] == 0x63) && (buf->data[7] == 0x73) && (buf->data[8] == 0x00) && (buf->data[9] == 0x03)) {
		mfg_data[8] = rssi;
		mfg_data[9] = 0x00;
	}
	if ((buf->data[6] == 0x63) && (buf->data[7] == 0x73) && (buf->data[8] == 0x00) && (buf->data[9] == 0x04)) {
		mfg_data[10] = rssi;
		mfg_data[11] = 0x00;
	}
	if ((buf->data[6] == 0x63) && (buf->data[7] == 0x73) && (buf->data[8] == 0x00) && (buf->data[9] == 0x05)) {
		mfg_data[12] = rssi;
		mfg_data[13] = 0x00;
	}
	if ((buf->data[6] == 0x63) && (buf->data[7] == 0x73) && (buf->data[8] == 0x00) && (buf->data[9] == 0x06)) {
		mfg_data[14] = rssi;
		mfg_data[15] = 0x00;
	}
	if ((buf->data[6] == 0x63) && (buf->data[7] == 0x73) && (buf->data[8] == 0x00) && (buf->data[9] == 0x07)) {
		mfg_data[16] = rssi;
		mfg_data[17] = 0x00;
	}
	if ((buf->data[6] == 0x63) && (buf->data[7] == 0x73) && (buf->data[8] == 0x00) && (buf->data[9] == 0x08)) {
		mfg_data[18] = rssi;
		mfg_data[19] = 0x00;
	}
}

void main(void) {
	struct bt_le_scan_param scan_param = {
		.type       = BT_HCI_LE_SCAN_PASSIVE,
		.options    = BT_LE_SCAN_OPT_NONE,
		.interval   = 0x0010,
		.window     = 0x0010,
	};
	int err;

	printk("Starting Scanner/Advertiser Demo\n");

	/* Initialize the Bluetooth Subsystem */
	err = bt_enable(NULL);
	if (err) {
		printk("Bluetooth init failed (err %d)\n", err);
		return;
	}

	printk("Bluetooth initialized\n");

	// scan for ads
	err = bt_le_scan_start(&scan_param, scan_cb);
	if (err) {
		printk("Starting scanning failed (err %d)\n", err);
		return;
	}

	// advertise
	do {
		k_sleep(K_MSEC(10));

		/* Start advertising */
		err = bt_le_adv_start(BT_LE_ADV_NCONN, ad, ARRAY_SIZE(ad),
				      NULL, 0);
		if (err) {
			printk("Advertising failed to start (err %d)\n", err);
			return;
		}

		k_sleep(K_MSEC(10));

		err = bt_le_adv_stop();
		if (err) {
			printk("Advertising failed to stop (err %d)\n", err);
			return;
		}
	} while (1);
}
